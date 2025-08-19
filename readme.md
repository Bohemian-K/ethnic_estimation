# Y-STR Ancestry Classification (OvR-MoE & OvO-MoE Pipelines)

This repository provides a Python implementation of One-vs-Rest (OvR) and One-vs-One (OvO) Mixture-of-Experts (MoE) pipelines for forensic ancestry inference using Y-STR markers.  
The pipeline integrates feature selection (`SelectOneFeaturePerMarker`), probability calibration (`platt_calibrate`), and an attention-based meta-learner (`AttentionMetaLearner`).

---

## Usage

1.Prepare Input Data
   Place your cleaned dataset in CSV format (default path: `./Data.csv`).  
   The dataset should include:
   - 20 Y-STR loci as features  
   - A column named `ethnic group` as the target label  

2.Run the Pipeline 
   ```bash
   python pipeline.py
````

3.Outputs

   * Cross-validation results comparing OvR-MoE and OvO-MoE pipelines
   * Summary report with performance metrics and confusion matrices (`./results/`)
   * Final trained OvR-MoE models and encoder saved in `./results/final_models/`



