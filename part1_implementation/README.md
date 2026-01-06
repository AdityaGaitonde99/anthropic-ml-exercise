# **Part 1 — Model Implementation & Training**

**SST-2 Sentiment Classification with Attention Visualization**

**Candidate:** Aditya Gaitonde

This directory contains my solution for **Part 1: Model Implementation & Training** of the Anthropic ML Engineer Technical Exercise (Option A).

The goal of this part is to implement, train, and analyze a transformer-based sentiment classifier, with emphasis on:

* correct data handling and evaluation

* interpretability via attention visualization

* error analysis and targeted edge-case slices

* clean, reproducible training code

---

## **Overview**

I fine-tune a transformer encoder (**DistilBERT**) on the **SST-2** dataset for binary sentiment classification.

This part includes:

* training and evaluation code

* trained model checkpoints

* quantitative metrics and calibration analysis

* qualitative attention visualizations

* lightweight ablation studies to justify design choices

---

## **Dataset and Split Strategy (Important)**

**Dataset:** `stanfordnlp/sst2` (Hugging Face)

The Hugging Face SST-2 **test split is unlabeled**, so it cannot be used for scoring.  
 To avoid data leakage while still enabling early stopping, I use the following setup:

* **HF train** → source for training

* **Internal dev split** → stratified split carved from HF train (`train_dev_split = 0.1`), used for early stopping and model selection

* **HF validation** → treated as the final **test set**, evaluated once at the end

This ensures:

* hyperparameter tuning and early stopping use dev only

* final metrics are reported on a held-out, labeled split

---

## **Directory Structure (Part 1\)**

`part1_implementation/`  
`├── model.py                 # Transformer classifier`  
`├── train.py                 # Training loop with early stopping`  
`├── evaluate.py              # Evaluation on held-out split`  
`├── data.py                  # Dataset loading & preprocessing`  
`├── viz.py                   # Attention visualization utilities`  
`├── ablation.py              # Ablation experiment runner`  
`├── config.yaml              # Base configuration`  
`├── config_final.yaml        # Locked configuration for final run`  
`├── README.md`                  
`├── report.md                # Technical report for Part 1`  
`├── requirements.txt`  
`└── outputs/`  
    `├── final_run/`  
    `│   ├── checkpoints/     # Trained model checkpoints`  
    `│   ├── metrics/         # Metrics, predictions, error analysis`  
    `│   └── figures/         # Plots and attention heatmaps`  
    `└── ablations/           # Outputs from ablation experiments`

---

## **Environment Setup**

* Python 3.10+

* PyTorch

* Hugging Face `transformers` and `datasets`

Install dependencies:

`pip install -r requirements.txt`

This code was tested primarily in **Google Colab**, but also runs locally.

---

## **Reproducing the Final Run**

The final, locked experiment configuration is defined in:

`config_final.yaml`

All outputs are written under:

`outputs/final_run/`

### **Training (with early stopping on dev)**

`python train.py \`  
  `--config config_final.yaml \`  
  `--device cuda \`  
  `--amp`

Key artifacts written:

* `outputs/final_run/checkpoints/best.pt`

* `outputs/final_run/checkpoints/last.pt`

* `outputs/final_run/checkpoints/best_infer.pt` (lightweight inference checkpoint)

* `outputs/final_run/metrics/train_history.json`

* `outputs/final_run/metrics/test_metrics_from_train.json`

---

### **Evaluation on Test Split (HF validation)**

Evaluation uses the lightweight inference checkpoint:

`python evaluate.py \`  
  `--config config_final.yaml \`  
  `--ckpt outputs/final_run/checkpoints/best_infer.pt \`  
  `--split test \`  
  `--device cuda`

Outputs written:

* `outputs/final_run/metrics/test_metrics.json`

* `outputs/final_run/metrics/test_predictions.csv`

* `outputs/final_run/metrics/test_misclassified.csv`

* `outputs/final_run/figures/*`

---

## **Results Summary (Final Run)**

Final scored split: **HF validation (872 examples)**

* **Accuracy:** 0.9014

* **F1:** 0.9042

* **Confusion matrix:** `[[380, 48], [38, 406]]`

* **Calibration:**

  * Brier score: 0.154

  * ECE (15 bins): 0.059

See `report.md` for a detailed discussion of metrics, slice-level behavior, attention analysis, and failure modes.

---

## **Attention Visualization**

Attention heatmaps are generated for targeted categories, including:

* negation

* contrast

* failure cases

These are saved under:

`outputs/final_run/figures/`

For readability:

* padding tokens are removed from axis labels

* special tokens may be excluded depending on configuration

---

## **Ablation Experiments**

A small set of ablations explores the effect of:

* pooling strategy

* dropout rate

Ablations reuse the same training pipeline with checkpoints disabled to remain lightweight.

Run ablations:

`python ablation.py \`  
  `--base_config config_final.yaml \`  
  `--device cuda \`  
  `--amp`

Results are summarized in:

`outputs/ablations/ablation_summary.json`

---

## **Notes**

* Checkpoints are included for reproducibility.

* The lightweight `best_infer.pt` checkpoint is sufficient for evaluation and analysis.

* This README focuses on **execution and structure**; modeling decisions, error analysis, and qualitative insights are discussed in `report.md`.

