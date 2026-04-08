# EEG Motor Imagery Classification with CSP and FBCSP

## 📌 Overview

This project implements a complete EEG classification pipeline for motor imagery tasks using:

* Common Spatial Pattern (CSP)
* Filter Bank CSP (FBCSP)
* Feature Selection (Mutual Information)
* Stratified K-Fold Cross Validation

The goal is to evaluate how multi-band spatial filtering improves classification performance.

---

## 🧠 Methods

### 1. CSP

Spatial filtering to maximize variance difference between classes.

### 2. FBCSP

EEG signals are decomposed into multiple frequency bands, and CSP is applied to each band.

### 3. Feature Selection

Mutual Information is used to select the most discriminative features.

---

## ⚙️ Experimental Setup

* Cross-validation: StratifiedKFold (5 splits, shuffle=True)
* CSP components: n = 6
* Frequency bands: 4–40 Hz (step = 4 Hz)

---

## 📊 Results

| Method     | Accuracy |
| ---------- | -------- |
| CSP        | XX%      |
| FBCSP      | XX%      |
| FBCSP + FS | XX%      |

---

## 🚀 How to Run

```bash
pip install -r requirements.txt
python experiments/run_fbcsp.py
```

---

## 📁 Project Structure

(see folders above)

---

## 📌 Future Work

* Real-time EEG classification
* Deep learning (EEGNet)
* Adaptive frequency band selection

---

## 👤 Author

[Your Name]

---

## ⭐ Notes

This project is built as a research-oriented implementation for EEG signal processing and machine learning.
# EEG-LDA-CSP-FBCSP
