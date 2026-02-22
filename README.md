# 🧠 EEG Seizure Detection

A machine learning pipeline for automated epilepsy detection from EEG signals using the **Bonn University Dataset**. The project classifies EEG recordings into three categories — Healthy, Interictal, and Ictal (seizure) — using a **1D CNN** and a **Bayesian-Optimized SVM**.

---

## 📁 Project Structure

```
eeg-seizure-prediction/
│
├── epilepsy_preprocessing.ipynb   # Data loading, visualization & feature extraction
├── epilepsy_cnn_model.ipynb       # 1D CNN + Bayesian SVM training & evaluation
│
├── dataset/                       # Bonn EEG dataset (not included)
│   ├── Z/                         # Set A — Healthy, eyes open
│   ├── O/                         # Set B — Healthy, eyes closed
│   ├── N/                         # Set C — Interictal, opposite hemisphere
│   ├── F/                         # Set D — Interictal, epileptogenic zone
│   └── S/                         # Set E — Ictal (active seizure)
│
├── X_train.npy                    # Preprocessed training features
├── X_test.npy                     # Preprocessed test features
├── y_train.npy                    # Training labels
├── y_test.npy                     # Test labels
├── preprocessing_pipeline.pkl     # Saved sklearn preprocessing pipeline
├── svm_epilepsy_model.pkl         # Saved SVM model
│
└── requirements.txt
```

---

## 📊 Dataset

**Bonn University EEG Dataset** — 5 sets (A–E), each containing 100 single-channel EEG segments.

| Set | Folder | Description | Label |
|-----|--------|-------------|-------|
| A | `Z` | Healthy volunteers, eyes open | 0 — Healthy |
| B | `O` | Healthy volunteers, eyes closed | 0 — Healthy |
| C | `N` | Epileptic patients, seizure-free (opposite hemisphere) | 1 — Interictal |
| D | `F` | Epileptic patients, seizure-free (epileptogenic zone) | 1 — Interictal |
| E | `S` | Epileptic patients, active seizure activity | 2 — Ictal |

- **Sampling rate:** 173.61 Hz  
- **Duration per segment:** ~23.6 seconds  
- **Total signals:** 500 (100 per set)

> Download the dataset from [Bonn University](https://www.ukbonn.de/epileptologie/arbeitsgruppen/ag-lehnertz-neurophysik/downloads/) and place it in the `dataset/` folder.

---

## 🔬 Pipeline Overview

### 1. Preprocessing (`epilepsy_preprocessing.ipynb`)

- **Data Loading** — reads `.txt` EEG files from all 5 folders
- **Visualization** — plots raw signals and statistical distributions per class
- **Bandpass Filtering** — Butterworth filter (0.5–60 Hz)
- **Normalization** — StandardScaler / MinMaxScaler
- **Variational Mode Decomposition (VMD)** — decomposes signals into intrinsic mode functions
- **Entropy Feature Extraction** — Shannon, sample, approximate, and permutation entropy
- **Feature Selection** — SelectKBest with mutual information and F-test
- **Dimensionality Reduction** — PCA
- **Class Imbalance Handling** — ADASYN oversampling

**Output:** `X_train.npy`, `X_test.npy`, `y_train.npy`, `y_test.npy`, `preprocessing_pipeline.pkl`

---

### 2. Model Training (`epilepsy_cnn_model.ipynb`)

#### Model 1 — 1D Convolutional Neural Network (CNN)

```
Conv1D(24, kernel=5) → BatchNorm → ReLU
Conv1D(16, kernel=3) → BatchNorm → ReLU → Dropout
Conv1D(8,  kernel=3) → BatchNorm → ReLU
GlobalAveragePooling
Dense(64) → ReLU → Dropout
Dense(3)  → Softmax
```

- Evaluated with accuracy, confusion matrix, and ROC curves (One-vs-Rest)

#### Model 2 — Bayesian-Optimized SVM

| Hyperparameter | Search Space | Prior |
|---|---|---|
| `C` | [0.01, 1000] | log-uniform |
| `gamma` | [1e-4, 10] | log-uniform |
| `kernel` | rbf, poly, sigmoid | categorical |
| `class_weight` | None, balanced | categorical |

- **Tuning:** BayesSearchCV — 40 iterations, 5-fold stratified CV
- **Scoring:** ROC-AUC (One-vs-Rest, macro)
- **Feature Importance:** Permutation importance on PCA components

Both models are compared side-by-side on the same test set.

---

## ⚙️ Setup

### 1. Clone the repository
```bash
git clone https://github.com/shruthi-nagappan/eeg-seizure-prediction.git
cd eeg-seizure-prediction
```

### 2. Create a virtual environment
```bash
python3 -m venv venv
source venv/bin/activate      # Mac/Linux
venv\Scripts\activate         # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Add the dataset
Download the Bonn dataset and place the folders (`Z`, `O`, `N`, `F`, `S`) inside a `dataset/` directory.

### 5. Run the notebooks in order
```
1. epilepsy_preprocessing.ipynb
2. epilepsy_cnn_model.ipynb
```

---

## 📦 Dependencies

- `numpy`, `pandas`, `matplotlib`, `seaborn`
- `scipy`, `vmdpy`, `antropy`
- `scikit-learn`, `imbalanced-learn`
- `tensorflow` / `keras`
- `scikit-optimize` (BayesSearchCV)

---

## 📈 Results

| Model | Test Accuracy | AUC (macro) |
|-------|--------------|-------------|
| 1D CNN | — | — |
| Bayesian SVM | — | — |

> Fill in your results after training.

---

## 👩‍💻 Author

**Shruthi Nagappan**  
[GitHub](https://github.com/shruthi-nagappan/eeg-seizure-prediction)
