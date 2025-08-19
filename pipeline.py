import pandas as pd
import numpy as np
from datetime import datetime
import os
import re
import joblib
import random
from itertools import combinations
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ML libs
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression, LassoCV
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, roc_auc_score, precision_recall_fscore_support)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.base import clone
from sklearn.tree import DecisionTreeClassifier
from skopt import gp_minimize
from skopt.space import Real, Integer
from joblib import Parallel, delayed
from collections import Counter
from sklearn.calibration import CalibratedClassifierCV

# SHAP (optional)
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    print("Warning: SHAP not available.")
    SHAP_AVAILABLE = False

# --- centralized config ---
CONFIG = {
    "input_data_path": './Data.csv',
    "target_column": 'ethnic group',
    "feature_columns": ['DYS19','DYS385a','DYS385b','DYS389I','DYS389II','DYS390','DYS391','DYS392','DYS393','DYS437','DYS438','DYS439','DYS448','DYS456','DYS458','DYS481','DYS533','DYS570','DYS576','DYS635','YGATAH4'],
    "iqr_multiplier": 1.5, "seed": 42, "bayesian_n_calls": 10, "cv_folds": 5, "test_cv_folds": 5,
    "n_jobs": -1, "moe_meta_learning_folds": 3, "sequential_cv_folds": 3,
    "lasso_alpha_range": [0.001, 0.01, 0.1, 1.0],
    "ensemble_weight_sequential": 0.4, "ensemble_weight_lasso": 0.3, "ensemble_weight_mutual_info": 0.3
}

def set_seed(seed):
    np.random.seed(seed); random.seed(seed); os.environ['PYTHONHASHSEED'] = str(seed)
set_seed(CONFIG['seed'])

# ==============================================================================
# Section 1: Data filtering and preprocessing
# ==============================================================================
def filter_data_by_group_iqr(df, group_col, feature_cols, iqr_multiplier):
    """Per-group IQR filter; removes multi-allelic rows and outliers per group."""
    print("IQR filtering per group...")
    filtered_dfs, original_count = [], len(df)
    for name, group in df.groupby(group_col):
        is_single = lambda v: pd.notna(v) and ',' not in str(v) and str(v).replace('.','',1).isdigit()
        single_allele_group = group[group[feature_cols].applymap(is_single).all(axis=1)]
        if single_allele_group.empty: continue
        numeric_group = single_allele_group.copy()
        numeric_group[feature_cols] = numeric_group[feature_cols].apply(pd.to_numeric)
        q1, q3 = numeric_group[feature_cols].quantile(0.25), numeric_group[feature_cols].quantile(0.75)
        iqr = q3 - q1; lower_bound, upper_bound = q1 - iqr_multiplier * iqr, q3 + iqr_multiplier * iqr
        final_group = numeric_group[((numeric_group[feature_cols] >= lower_bound) & (numeric_group[feature_cols] <= upper_bound)).all(axis=1)]
        filtered_dfs.append(final_group)
    if not filtered_dfs:
        print("Error: empty data after filtering.")
        return None
    final_df = pd.concat(filtered_dfs, ignore_index=True)
    print(f"Filtering done. Kept {len(final_df)} of {original_count} rows.")
    return final_df

def one_hot_encode_features(df, target_col):
    """One-hot for all STR loci (keep microvariants as strings)."""
    features, target = df.drop(columns=[target_col]).astype(str), df[target_col]
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    features_encoded_np = encoder.fit_transform(features)
    clean_name = lambda name: re.sub(r'[^A-Za-z0-9_]+', '_', name)
    feature_names = [clean_name(col) for col in encoder.get_feature_names_out(features.columns)]
    return pd.DataFrame(features_encoded_np, columns=feature_names).astype(bool), target, encoder

# ==============================================================================
# Section 2: 4-stage hybrid allele choice system
# ==============================================================================
class SelectOneFeaturePerMarker:
    def __init__(self, config): 
        self.config, self.selection_details = config, {}
    def select_alleles_for_full_profile(self, X, y, marker_names):
        print("  - Selecting one feature per marker (hybrid ensemble scoring)")
        final_alleles = []
        for marker in marker_names:
            candidates = [col for col in X.columns if col.startswith(marker + '_')]
            if not candidates: continue
            scores = {
                'seq': self._score_by_sequential(X, y, candidates),
                'lasso': self._score_by_lasso(X, y, candidates),
                'mi': self._score_by_mi(X, y, candidates)
            }
            best_allele, best_score, details = self._ensemble_choice(X, candidates, scores)
            final_alleles.append(best_allele)
            self.selection_details[marker] = {'chosen_allele': best_allele, 'ensemble_score': best_score, 'details': details}
        return final_alleles
    def _score_by_sequential(self, X, y, alleles): 
        return {a: self._evaluate_combination(X[[a]], y) for a in alleles}
    def _score_by_lasso(self, X, y, alleles):
        if len(alleles) < 2: return {a: 0.1 for a in alleles}
        try:
            l = LassoCV(alphas=self.config['lasso_alpha_range'], cv=self.config['sequential_cv_folds'], random_state=self.config['seed']).fit(X[alleles], y)
            return {a: np.abs(l.coef_[i]) for i, a in enumerate(alleles)}
        except:
            return {a: 0.0 for a in alleles}
    def _score_by_mi(self, X, y, alleles): 
        return {a: mi for a, mi in zip(alleles, mutual_info_classif(X[alleles], y, random_state=self.config['seed']))}
    def _ensemble_choice(self, X, alleles, scores):
        ensemble_scores = Counter()
        max_s = {k: max(v.values()) if v else 1.0 for k, v in scores.items()}
        for a in alleles:
            norm_seq, norm_lasso, norm_mi = scores['seq'].get(a, 0), scores['lasso'].get(a, 0) / (max_s['lasso'] or 1.0), scores['mi'].get(a, 0) / (max_s['mi'] or 1.0)
            ensemble_scores[a] = self.config['ensemble_weight_sequential']*norm_seq + self.config['ensemble_weight_lasso']*norm_lasso + self.config['ensemble_weight_mutual_info']*norm_mi
        best_allele = ensemble_scores.most_common(1)[0][0] if ensemble_scores else X[alleles].sum().idxmax()
        return best_allele, ensemble_scores[best_allele], {a: ensemble_scores[a] for a in alleles}
    def _evaluate_combination(self, X_feat, y):
        y_enc = LabelEncoder().fit_transform(y)
        if len(np.unique(y_enc)) < 2: return 0.5
        cv = StratifiedKFold(n_splits=self.config['sequential_cv_folds'], shuffle=True, random_state=self.config['seed'])
        evaluators = {'RF': RandomForestClassifier(n_estimators=20, max_depth=5), 'XGB': XGBClassifier(n_estimators=20, max_depth=3, use_label_encoder=False, eval_metric='logloss'), 'LR': LogisticRegression()}
        weights = {'RF': 0.4, 'XGB': 0.4, 'LR': 0.2}
        fold_scores = []
        for train_idx, val_idx in cv.split(X_feat, y_enc):
            X_train, X_val, y_train, y_val = X_feat.iloc[train_idx], X_feat.iloc[val_idx], y_enc[train_idx], y_enc[val_idx]
            if len(np.unique(y_train)) < 2: continue
            score = sum(roc_auc_score(y_val, clone(m).set_params(random_state=self.config['seed']).fit(X_train, y_train).predict_proba(X_val)[:, 1]) * w for m, w in zip(evaluators.values(), weights.values()))
            fold_scores.append(score)
        return np.mean(fold_scores) if fold_scores else 0.5

# ==============================================================================
# Platt calibration wrapper
# ==============================================================================
def platt_calibrate(clf, X, y, method="sigmoid", cv=3):
    cal = CalibratedClassifierCV(clf, method=method, cv=cv)
    cal.fit(X, y)
    return cal

# ==============================================================================
# Section 3: 2-Level Mixture of Experts (MoE) models
# ==============================================================================
class AttentionMetaLearner:
    """Binary MoE per task (original MicroMoE)."""
    def __init__(self, experts, random_state=42):
        self.experts, self.names = experts, list(experts.keys())
        self.gate = LogisticRegression(random_state=random_state)
        self.perf, self.le = {}, LabelEncoder()
    def fit(self, X, y):
        y_enc = self.le.fit_transform(y)
        for name, expert in self.experts.items():
            expert.fit(X, y_enc)
            try: self.perf[name] = roc_auc_score(y_enc, expert.predict_proba(X)[:, 1])
            except ValueError: self.perf[name] = 0.5
        self._train_gate(X, y_enc); return self
    def _train_gate(self, X, y_enc):
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42); meta_X, meta_y = [], []
        for tr_idx, val_idx in cv.split(X, y_enc):
            X_tr, X_val, y_tr, y_val = X.iloc[tr_idx], X.iloc[val_idx], y_enc[tr_idx], y_enc[val_idx]
            if len(np.unique(y_tr)) < 2: continue
            preds = {n: clone(m).fit(X_tr, y_tr).predict_proba(X_val)[:, 1] for n, m in self.experts.items()}
            for i, (idx, row) in enumerate(X_val.iterrows()):
                s_preds = np.array([preds[n][i] for n in self.names])
                meta_y.append(np.argmax(s_preds) if y_val[i] == 1 else np.argmin(s_preds))
                meta_X.append(np.concatenate([row.describe().values, s_preds, list(self.perf.values())]))
        if len(np.unique(meta_y)) > 1: self.gate.fit(np.array(meta_X), np.array(meta_y))
        else: self.gate = None
    def predict_proba(self, X):
        expert_preds = np.array([m.predict_proba(X)[:, 1] for m in self.experts.values()]).T
        if self.gate is None: weights = np.full_like(expert_preds, 1/expert_preds.shape[1])
        else:
            meta_X_pred = np.array([np.concatenate([row.describe().values, expert_preds[i], list(self.perf.values())]) for i, (idx, row) in enumerate(X.iterrows())])
            weights = self.gate.predict_proba(meta_X_pred)
        final_preds = np.clip(np.sum(expert_preds * weights, axis=1), 0, 1)
        return np.column_stack([1 - final_preds, final_preds])
    def predict(self, X): 
        return self.le.inverse_transform(np.argmax(self.predict_proba(X), axis=1))

class OvR_MacroMoE:
    """Multiclass gating over OvR experts (kept original class name to avoid collision)."""
    def __init__(self, experts):
        self.experts, self.names = experts, sorted(experts.keys())
        self.gate = LogisticRegression(multi_class='multinomial')
        self.le = LabelEncoder()
    def fit(self, X, y):
        print("  - Training OvR macro-level gating...")
        self.gate.fit(self._get_meta(X), self.le.fit_transform(y))
        return self
    def _get_meta(self, X):
        return pd.DataFrame({n: i['model'].predict_proba(X[i['selected_features']])[:, 1] for n, i in self.experts.items()})
    def predict(self, X):
        return self.le.inverse_transform(np.argmax(self.gate.predict_proba(self._get_meta(X)), axis=1))

class OvO_VotingSystem:
    """One-vs-One majority voting."""
    def __init__(self, experts):
        self.experts = experts
    def predict(self, X):
        predictions = []
        for i in range(len(X)):
            instance_votes = Counter()
            for pair_info in self.experts.values():
                pred = pair_info['model'].predict(X[pair_info['selected_features']].iloc[[i]])[0]
                instance_votes.update([pred])
            predictions.append(instance_votes.most_common(1)[0][0])
        return pd.Series(predictions)

# ==============================================================================
# Section 4: Model building and evaluation pipeline (OvR + OvO)
# ==============================================================================
BASE_MODELS = {
    'LR': {'m': LogisticRegression(C=1.0), 'p': [Real(0.01, 100, name='C')]},
    'RF': {'m': RandomForestClassifier(), 'p': [Integer(50, 150, name='n_estimators'), Integer(5, 20, name='max_depth')]}
}

def build_expert_model(task_id, X, y, config, model_type='OvR'):
    """Build a single expert for OvR (binary) or OvO (pairwise) using the selector."""
    if model_type == 'OvR':
        nation, y_target, X_target = task_id, (y == task_id).astype(int), X
    else:
        nation, pair = f"{task_id[0]}_vs_{task_id[1]}", task_id
        y_target = y[y.isin(pair)]
        X_target = X.loc[y_target.index]
    print(f"--- Building expert: {model_type} -> '{nation}' ---")
    if len(y_target.unique()) < 2:
        print("Warning: only one class present. Skipping.")
        return task_id, None

    selector = SelectOneFeaturePerMarker(config)
    feats = selector.select_alleles_for_full_profile(X_target, y_target, config['feature_columns'])
    X_sel, y_target_sel = X_target[feats].reset_index(drop=True), y_target.reset_index(drop=True)

    # Bayesian tuning and expert training
    experts = {}
    for name, model_info in BASE_MODELS.items():
        def obj_func(p):
            params = {pn: v for pn, v in zip([d.name for d in model_info['p']], p)}
            model = clone(model_info['m']).set_params(**params)
            scores = cross_val_score(model, X_sel, y_target_sel, cv=config['cv_folds'])
            return 1.0 - np.mean(scores)
        res = gp_minimize(obj_func, model_info['p'], n_calls=config['bayesian_n_calls'], random_state=config['seed'])
        tuned = clone(model_info['m']).set_params(**{pn: v for pn, v in zip([d.name for d in model_info['p']], res.x)})
        experts[name] = tuned

    # Expert-level MoE (AttentionMetaLearner)
    model = AttentionMetaLearner(experts, random_state=config['seed']).fit(X_sel, y_target_sel)

    return task_id, {
        'model': model,
        'selected_features': feats,
        'selection_details': selector.selection_details,
        'internal_performance': model.perf
    }

def comprehensive_evaluation(X, y, config):
    """Cross-validated comparison: OvR 2-level MoE vs OvO 2-level MoE."""
    print("\n" + "="*80 + "\nComprehensive CV evaluation: OvR-MoE vs OvO-MoE\n" + "="*80)
    cv = StratifiedKFold(n_splits=config['test_cv_folds'], shuffle=True, random_state=config['seed'])
    results = {n: {'true': [], 'pred': []} for n in ['OvR_2-Level_MoE', 'OvO_2-Level_MoE']}
    viz_dir = f"./results/moe_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    for d in ['OvR_Experts', 'OvO_Experts', 'OvR_2-Level_MoE', 'OvO_2-Level_MoE']:
        os.makedirs(f"{viz_dir}/{d}", exist_ok=True)

    last_fold_experts = {}
    for fold, (train_idx, test_idx) in enumerate(cv.split(X, y), start=1):
        print(f"\n--- Fold {fold}/{config['test_cv_folds']} ---")
        X_train, X_test, y_train, y_test = X.iloc[train_idx], X.iloc[test_idx], y.iloc[train_idx], y.iloc[test_idx]
        nations = sorted(y_train.unique())

        # Pipeline A: OvR-based
        ovr_res = Parallel(n_jobs=config['n_jobs'])(delayed(build_expert_model)(n, X_train, y_train, config, 'OvR') for n in nations)
        clf_ovr = {res[0]: res[1] for res in ovr_res if res[1] is not None}
        if clf_ovr:
            ovr_macro_moe = OvR_MacroMoE(experts=clf_ovr).fit(X_train, y_train)
            pred_ovr = ovr_macro_moe.predict(X_test)
            results['OvR_2-Level_MoE']['true'].extend(y_test)
            results['OvR_2-Level_MoE']['pred'].extend(pred_ovr)
            print(f"OvR_2-Level_MoE Fold {fold} - Accuracy: {accuracy_score(y_test, pred_ovr):.4f}")

        # Pipeline B: OvO-based
        ovo_pairs = list(combinations(nations, 2))
        ovo_res = Parallel(n_jobs=config['n_jobs'])(delayed(build_expert_model)(p, X_train, y_train, config, 'OvO') for p in ovo_pairs)
        clf_ovo = {res[0]: res[1] for res in ovo_res if res[1] is not None}
        if clf_ovo:
            ovo_voting_system = OvO_VotingSystem(experts=clf_ovo)
            pred_ovo = ovo_voting_system.predict(X_test)
            results['OvO_2-Level_MoE']['true'].extend(y_test)
            results['OvO_2-Level_MoE']['pred'].extend(pred_ovo)
            print(f"OvO_2-Level_MoE Fold {fold} - Accuracy: {accuracy_score(y_test, pred_ovo):.4f}")

        if fold == config['test_cv_folds'] - 1:
            last_fold_experts = {'ovr': clf_ovr, 'ovo': clf_ovo}

        if fold == 0:
            if clf_ovr:
                try:
                    visualize_chosen_alleles_heatmap(clf_ovr, f"{viz_dir}/OvR_Experts/alleles_heatmap.png")
                except Exception:
                    pass
            if clf_ovo:
                try:
                    visualize_chosen_alleles_heatmap({f'{k[0]}-{k[1]}':v for k,v in clf_ovo.items()}, f"{viz_dir}/OvO_Experts/alleles_heatmap.png")
                except Exception:
                    pass

    return viz_dir, results, last_fold_experts

def generate_summary_report(viz_dir, results, experts, config):
    """Write text summary report; create confusion matrices if helper exists."""
    report_path = f"{viz_dir}/summary_report.txt"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + f"\nY-STR Ancestry Classification Final Report\nRun: {datetime.now():%Y-%m-%d %H:%M:%S}\n" + "="*80 + "\n\n1) Pipeline Performance (all folds)\n" + "-"*40 + f"\n{'Pipeline':<20} | {'Accuracy':<10} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10}\n" + "-"*70 + "\n")
        labels = sorted(np.unique(results['OvR_2-Level_MoE']['true'] + results['OvR_2-Level_MoE']['pred']))
        for name, res in results.items():
            if not res['true']: continue
            acc = accuracy_score(res['true'], res['pred'])
            p, r, f1, _ = precision_recall_fscore_support(res['true'], res['pred'], average='weighted', labels=labels, zero_division=0)
            f.write(f"{name:<20} | {acc:<10.4f} | {p:<10.4f} | {r:<10.4f} | {f1:<10.4f}\n")

        f.write("\n\n2) Detailed Analysis (all folds combined)\n" + "-"*40)
        for name, res in results.items():
            if not res['true']: continue
            f.write(f"\n\n--- Pipeline: {name} ---\nClassification report:\n{classification_report(res['true'], res['pred'], labels=labels, zero_division=0)}\nConfusion matrix:\n{pd.DataFrame(confusion_matrix(res['true'], res['pred'], labels=labels), index=[f'True:{l}' for l in labels], columns=[f'Pred:{l}' for l in labels]).to_string()}\n")
            try:
                create_confusion_matrix(res['true'], res['pred'], labels, f"{name} (All Folds)", f"{viz_dir}/{name}_cm_all.png")
            except Exception:
                pass

        for exp_type, exp_dict in experts.items():
            if not exp_dict: continue
            title = "OvR (One-vs-Rest)" if exp_type == 'ovr' else "OvO (One-vs-One)"
            f.write(f"\n\n3) {title} Expert Models (last fold)\n" + "-"*60 + "\n")
            for task_id, info in exp_dict.items():
                name = task_id if isinstance(task_id, str) else f"{task_id[0]}_vs_{task_id[1]}"
                f.write(f"\n--- Expert: {name} ---\n")
                f.write("  - Internal experts (AUC):\n")
                for sub_exp, perf in info['internal_performance'].items():
                    f.write(f"    - {sub_exp:<5}: {perf:.4f}\n")
                f.write("  - Selected allele profile:\n")
                for marker, details in info['selection_details'].items():
                    f.write(f"    - {marker:<10}: {details['chosen_allele']} (Score: {details['ensemble_score']:.3f})\n")
    print(f"Summary report saved: {report_path}")

# ==============================================================================
# Section 5: Main
# ==============================================================================
def main():
    print("Y-STR pipeline start (v5.1; names aligned to PDF terms)")
    if not os.path.exists(CONFIG['input_data_path']):
        print(f"Error: {CONFIG['input_data_path']} not found.")
        return

    raw_data = pd.read_csv(CONFIG['input_data_path'])
    filtered_data = filter_data_by_group_iqr(raw_data, CONFIG['target_column'], CONFIG['feature_columns'], CONFIG['iqr_multiplier'])
    if filtered_data is None:
        return

    X_final, y_final, encoder = one_hot_encode_features(filtered_data, CONFIG['target_column'])

    viz_dir, results, last_fold_experts = comprehensive_evaluation(X_final, y_final, CONFIG)
    generate_summary_report(viz_dir, results, last_fold_experts, CONFIG)

    print("\n" + "="*80 + "\nTrain final OvR models on full data\n" + "="*80)
    # Final OvR models
    final_ovr_res = Parallel(n_jobs=CONFIG['n_jobs'])(delayed(build_expert_model)(n, X_final, y_final, CONFIG, 'OvR') for n in sorted(y_final.unique()))
    final_clf_ovr = {res[0]: res[1] for res in final_ovr_res if res[1] is not None}
    if final_clf_ovr:
        final_ovr_macro_moe = OvR_MacroMoE(experts=final_clf_ovr).fit(X_final, y_final)
        model_dir = f"{viz_dir}/final_models"; os.makedirs(model_dir, exist_ok=True)
        joblib.dump(final_clf_ovr, f"{model_dir}/ovr_level1_experts.joblib")
        joblib.dump(final_ovr_macro_moe, f"{model_dir}/ovr_level2_macro_moe.joblib")
        joblib.dump(encoder, f"{model_dir}/encoder.joblib")
        print(f"Final OvR-MoE models saved to '{model_dir}'.")

    print("\nDone.")

if __name__ == '__main__':
    main()
