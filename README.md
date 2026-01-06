# **Anthropic ML Engineer Technical Exercise**

**Candidate:** Aditya Gaitonde

This repository contains my complete submission for the **Anthropic Machine Learning Engineer Technical Exercise**, including **Part 1 (Model Implementation & Training)**, **Part 2 (Research Analysis)**, **Part 3 (Code Review)**, and the **optional bonus analysis**.

Each part is organized into a dedicated directory and can be reviewed independently.

---

## **Submission Overview**

* **Part 1 — Model Implementation & Training**  
   A transformer-based sentiment classifier trained on the SST-2 dataset. This part includes full training and evaluation code, trained checkpoints, quantitative metrics, attention visualizations, error analysis, and ablation studies.

* **Part 2 — Research / Analysis**  
   A written technical analysis discussing model reliability, evaluation challenges, and mitigation strategies.

* **Part 3 — Code Review Exercise**  
   A structured review of a provided transformer implementation, along with an improved version addressing correctness, robustness, and architectural concerns.

* **Bonus — Scaling Law Analysis (Optional)**  
   An exploratory analysis applying empirical scaling laws to reason about tradeoffs between model size, dataset size, and compute budget.

---

## **Repository Structure**
```text

anthropic-ml-exercise/
├── README.md
├── requirements.txt
├── part1_implementation/
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   ├── data.py
│   ├── viz.py
│   ├── ablation.py
│   ├── config.yaml
│   ├── config_final.yaml
│   ├── README.md
│   ├── report.md
│   └── outputs/
│       ├── final_run/
│       └── ablations/
├── part2_research_analysis/
│   └── technical_document.pdf
├── part3_code_review/
│   ├── review.md
│   └── improved_code.py
└── bonus_scaling_laws/
    └── scaling_analysis.ipynb

```
---

## **How to Review Each Part**

### **Part 1 — Model Implementation & Training**

* **Location:** `part1_implementation/`

* **Code:** `part1_implementation/`

* **Report:** `part1_implementation/report.md`

* **Outputs:** `part1_implementation/outputs/`

The outputs directory contains trained model checkpoints, evaluation metrics, misclassified examples, calibration plots, slice-level analysis, and attention visualizations for targeted edge cases (e.g., negation and contrast).

Detailed training and evaluation instructions, configuration details, and design assumptions are documented in the Part 1 README.

---

### **Part 2 — Research / Analysis**

* **Location:** `part2_research_analysis/technical_document.pdf`

This document analyzes model failure modes and evaluation concerns, proposes mitigation strategies, and discusses broader implications for deployment.

---

### **Part 3 — Code Review**

* **Location:** `part3_code_review/`

* **Review:** `review.md`

* **Improved Code:** `improved_code.py`

This part demonstrates reasoning about code quality, architectural correctness, and robustness in transformer-based implementations.

---

### **Bonus — Scaling Laws (Optional)**

* **Location:** `bonus_scaling_laws/scaling_analysis.ipynb`

This notebook explores empirical scaling laws and uses them to reason about optimal allocation of compute across model size and dataset size.

---

## **Execution Notes**

* **Part 1** is fully runnable and includes trained checkpoints for reproducibility and evaluation.

* **Parts 2 and 3** are analytical and review-based artifacts and are not intended to be executed.

* **Bonus** analysis is self-contained within the provided notebook.

---
## Full Submission Archive (Includes Checkpoints)

Due to GitHub file size limits, trained model checkpoints and large output artifacts are not stored directly in this repository.

Full archive (Part 1 outputs + checkpoints, plus Parts 2–3 and bonus): [https://drive.google.com/file/d/1SuRpsahK3vmro9wARD7jyzk1O5pUWkqX/view?usp=sharing]

Thank you for taking the time to review this submission.  
 I’m happy to clarify any details or discuss design decisions further.
 

