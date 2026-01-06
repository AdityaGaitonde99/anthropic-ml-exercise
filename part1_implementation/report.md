# **Part 1 Report — SST-2 Sentiment Classification with Attention Visualization**

**Candidate:** Aditya Gaitonde

## **1\. Objective**

Build and train a transformer-based sentiment classifier for SST-2, with a focus on:

1. clean, reproducible training/evaluation,

2. interpretable analysis via attention visualization, and

3. model behavior analysis on challenging “edge case” categories (negation, contrast, and failure cases), supported by quantitative and qualitative evidence.

---

## **2\. Dataset and Split Strategy (Leakage-Safe)**

**Dataset:** `stanfordnlp/sst2` (Hugging Face)

* Text column: `sentence`

* Label column: `label` (0 \= negative, 1 \= positive)

**Important SST-2 note:** The Hugging Face SST-2 **test** split is unlabeled, so it cannot be used for scoring. To avoid leakage while still enabling early stopping, I used:

* **HF train** → training source

* **Internal dev** → stratified split from HF train (`train_dev_split = 0.1`) for early stopping / model selection

* **HF validation** → treated as the final **test** split (scored), evaluated at the end

This ensures hyperparameter choices are based on dev only, while final reporting uses a held-out scored split.

---

## **3\. Model**

**Base encoder:** `distilbert-base-uncased`  
 **Task head:** binary classifier over pooled representation

**Key modeling choices**

* **Max sequence length:** 128

* **Pooling:** CLS pooling (`pooling = cls`)

* **Dropout:** 0.1

* Output is a standard two-class prediction with confidence/probabilities recorded for analysis.

---

## **4\. Training Procedure**

**Optimization and regularization**

* Batch size: 32 (train), 128 (eval)

* Learning rate: 2e-5

* Weight decay: 0.01

* Warmup ratio: 0.06

* Gradient clipping: 1.0

* Label smoothing: 0.0

**Early stopping**

* Best checkpoint chosen using **dev loss**

* Training ran for **3 epochs**, with best dev loss observed at **epoch 2** (see training dynamics below).

**Runtime**

* Total training runtime recorded: \~**421 seconds** (\~7 minutes) for the final run.

---

## **5\. Evaluation Protocol**

**Final scored split:** HF validation (**872** examples)  
 I report:

* Accuracy, F1

* Confusion matrix and classification report

* Calibration metrics (Brier score, ECE)

* Slice-level performance (negation / contrast / length buckets)

* Error analysis from misclassified examples

* Attention visualization outputs for targeted examples

---

## **6\. Quantitative Results (Final Run)**

### **6.1 Overall performance (HF validation, n=872)**

* **Accuracy:** 0.9014

* **F1:** 0.9042

* **Confusion matrix:** `[[380, 48], [38, 406]]`

Interpretation:

* Errors are relatively balanced, with a slightly higher number of **false positives (48)** than **false negatives (38)**.

### **6.2 Calibration**

* **Brier score:** 0.1543

* **ECE (15 bins):** 0.0594

Interpretation:

* ECE around \~0.06 suggests **reasonably good calibration**, but the error analysis (below) shows some **high-confidence mistakes** on harder linguistic constructions.

---

## **7\. Training Dynamics (from training curves)**

The training figures saved (`training_loss.png`, `training_acc.png`) are consistent with the recorded dev trajectory:

| Epoch | Dev Loss | Dev Accuracy |
| ----- | ----- | ----- |
| 1 | 0.1939 | 0.9265 |
| 2 | **0.1698** | 0.9399 |
| 3 | 0.1869 | **0.9411** |

Key observations:

* **Dev loss improves strongly from epoch 1 → 2**, then increases at epoch 3\.

* **Dev accuracy continues to rise slightly**, but the dev loss increase suggests **mild overfitting beginning after epoch 2**.

* Selecting the best checkpoint by dev loss (epoch 2\) is therefore well-motivated.

---

## **8\. Slice and Edge-Case Analysis**

A key goal of this exercise was to go beyond aggregate metrics and test behavior on linguistically challenging subsets.

### **8.1 Negation slice**

* **Negation \= True (n=189):** accuracy 0.8836, F1 0.8493

* **Negation \= False (n=683):** accuracy 0.9063, F1 0.9149

Observation:

* Performance drops on negation-heavy examples, which is expected: negation often flips sentiment and can interact with clause structure in non-local ways.

### **8.2 Contrast slice (e.g., “X, but Y”)**

* **Contrast \= True (n=130):** accuracy 0.8538, F1 0.8480

* **Contrast \= False (n=742):** accuracy 0.9097, F1 0.9133

Observation:

* Contrast cases are the hardest slice here (largest drop vs overall). These examples often require the model to correctly weight sentiment-bearing content across clauses, typically prioritizing the clause after “but/however/although”.

### **8.3 Length buckets**

| Length bucket | n | Accuracy | F1 |
| ----- | ----- | ----- | ----- |
| \<=4 | 5 | 0.8000 | 0.6667 |
| 5–8 | 53 | 0.9245 | 0.9000 |
| 9–16 | 196 | 0.8980 | 0.9010 |
| 17–32 | 447 | 0.9038 | 0.9091 |
| 33+ | 171 | 0.8947 | 0.9000 |

Observation:

* Very short inputs (\<=4) are rare and noisy.

* The model is fairly stable across typical sentence lengths; performance dips slightly at 33+ tokens, which may reflect increased compositional complexity rather than pure length.

---

## **9\. Attention Visualization (Qualitative Interpretability)**

I generated attention heatmaps for targeted categories and saved them under:

* `outputs/final_run/figures/` (examples labeled by category: `negation`, `contrast`, and `failure`)

Goal:

* Provide a concrete, inspectable view of token-level focus patterns for challenging examples.

How I used these visualizations:

* **Negation examples:** check whether attention highlights negation markers and whether sentiment-bearing adjectives/verbs are attended to appropriately.

* **Contrast examples:** verify whether the model focuses on the **dominant clause** (often the clause after contrast markers).

* **Failure cases:** identify patterns where attention concentrates on superficially positive tokens while ignoring negating/qualifying context (or vice versa).

These qualitative checks complement slice metrics by giving a readable hypothesis for *why* some constructions are harder.

---

## **10\. Error Analysis (Misclassified Examples)**

The misclassified set contains **86** examples. A notable pattern is the presence of **very high-confidence mistakes**, especially in contrast-heavy and negation-heavy sentences.

Two representative examples:

### **10.1 High-confidence false positives (true negative → predicted positive)**

Example (contrast-heavy, long):

* True: 0, Pred: 1, Confidence ≈ 0.999

* Sentence begins negatively (“clumsy”, “lethargically paced”) but the model predicts positive with extremely high confidence.

Interpretation:

* These are the exact cases where **calibration and clause-level reasoning** matter. Even with decent overall ECE, the model can still be overconfident on specific difficult constructions.

### **10.2 High-confidence false negatives (true positive → predicted negative)**

Example (short, potentially sarcastic/ambiguous):

* True: 1, Pred: 0, Confidence ≈ 0.997

* Sentence: “hilariously inept and ridiculous.”

Interpretation:

* Some SST-2 labels can be subjective depending on context and sarcasm. Phrases that appear negative (“inept”, “ridiculous”) can be used positively in certain reviewing styles. These cases are useful as “stress tests” for semantics beyond keyword polarity.

---

## **11\. Ablation Studies**

I ran lightweight ablations to validate key design choices and avoid relying on a single configuration.

From `outputs/ablations/ablation_summary.json`:

| Ablation | Test Acc | Delta vs Final |
| ----- | ----- | ----- |
| pooling\_cls | 0.9014 | \+0.00 pp |
| pooling\_mean | 0.8968 | \-0.46 pp |
| dropout\_0.1 | 0.9014 | \+0.00 pp |
| dropout\_0.0 | **0.9025** | \+0.11 pp |

Interpretation:

* **CLS pooling** outperformed mean pooling in this setup.

* **Dropout 0.0** yields a tiny accuracy gain (+0.11 pp), but the improvement is small enough that I kept dropout=0.1 as a reasonable regularization default (and to avoid brittle overfitting on dev).

* The ablations reinforce that the final configuration is competitive and not accidental.

---

## **12\. Limitations and Next Steps**

**Limitations**

* Contrast and negation remain consistent failure modes compared to the overall distribution.

* Some errors appear tied to subjective phrasing and label ambiguity (common in sentiment datasets).

* Calibration is decent overall, but there are still pockets of overconfidence in hard slices.

**Next steps (if extending this)**

* Add slice-aware training/evaluation (explicit “contrast marker” tag training diagnostics).

* Explore calibration improvements: temperature scaling on dev, or confidence penalties.

* Add richer interpretability: attention rollout or integrated gradients for stability vs single-layer attention snapshots.

---

## **13\. Reproducibility and Artifacts**

**Code:** `part1_implementation/`  
 **Final config:** `part1_implementation/config_final.yaml`  
 **Final outputs:** `outputs/final_run/`

* Checkpoints: `best.pt`, `last.pt`, `best_infer.pt`

* Metrics: `test_metrics.json`, `test_predictions.csv`, `test_misclassified.csv`

* Figures: training curves, confusion matrix, calibration plots, slice accuracy plots, and attention heatmaps

