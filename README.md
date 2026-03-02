# 🧠 EEG Seizure Detection

A machine learning pipeline for automated epilepsy detection from EEG signals using the **Bonn University Dataset**. The project classifies EEG recordings into three categories — Healthy, Interictal, and Ictal (seizure) — using a **1D CNN**, a **Bayesian-Optimized SVM**, and a **Transformer**.

---

## 📁 Project Structure

```
eeg-seizure-prediction/
│
├── epilepsy_preprocessing.ipynb      # Data loading, visualization & feature extraction
├── epilepsy_cnn_model.ipynb          # 1D CNN + Bayesian SVM training & evaluation
├── eeg_transformer_comparison.ipynb  # Transformer training & full 3-model comparison
│
├── dataset/                       # Bonn EEG dataset (not included)
│   ├── Z/                         # Set A — Healthy, eyes open
│   ├── O/                         # Set B — Healthy, eyes closed
│   ├── N/                         # Set C — Interictal, opposite hemisphere
│   ├── F/                         # Set D — Interictal, epileptogenic zone
│   └── S/                         # Set E — Ictal (active seizure)
│
├── X_train.npy                    # Preprocessed training features (CNN / SVM)
├── X_test.npy                     # Preprocessed test features (CNN / SVM)
├── y_train.npy                    # Training labels
├── y_test.npy                     # Test labels
├── preprocessing_pipeline.pkl     # Saved sklearn preprocessing pipeline
│
├── epilepsy_cnn_model.keras       # Saved CNN model
├── transformer_eeg_model.keras    # Saved Transformer model
├── checkpoints/
│   ├── best_model.keras           # Best CNN checkpoint
│   └── best_transformer.keras     # Best Transformer checkpoint
│
├── svm_epilepsy_model.pkl         # Saved SVM model
├── svm_best_params.json           # Best SVM hyperparameters from Bayesian search
│
├── all_models_comparison.png      # CNN vs SVM vs Transformer comparison dashboard
├── transformer_training_curves.png
├── transformer_evaluation.png
├── transformer_attention_map.png
├── per_class_f1_comparison.png
├── model_comparison_results.json
│
├── svm_convergence.png
├── svm_confusion_matrix.png
├── svm_roc_curves.png
├── svm_feature_importance.png
├── cnn_vs_svm_comparison.png
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

---

### 3. Transformer (`eeg_transformer_comparison.ipynb`)

#### Model 3 — Transformer Encoder on Raw EEG

Unlike the CNN and SVM which operate on hand-crafted entropy/PCA features, the Transformer is trained directly on **raw EEG windows**, allowing it to learn temporal patterns without manual feature engineering.

**Why raw signals?** Self-attention on ~20 PCA features has almost nothing meaningful to attend over. The Transformer's strength is capturing long-range temporal dependencies across the full waveform.

**Input pipeline:**
- Raw `.txt` files loaded from `dataset/` folders
- Butterworth bandpass filter (0.5–60 Hz)
- Z-score normalisation per segment
- Sliced into non-overlapping **256-sample windows** (~1.5 s each), yielding ~8,000 samples

**Architecture:**
```
Raw EEG window  (256 timesteps × 1 channel)
  └─► Conv1D(64, kernel=7) + LayerNorm    ← local patch embedding
      └─► Learnable Positional Encoding
          └─► TransformerBlock × 4
              ├─ MultiHeadAttention (4 heads, key_dim=16)
              ├─ Add & LayerNorm
              ├─ FFN  Dense(256 → 64)
              └─ Add & LayerNorm
          └─► GlobalAveragePooling1D
              └─► Dense(64, relu) → Dropout(0.2) → Dense(3, softmax)
```

- **Optimizer:** Adam (lr=5e-4)
- **Callbacks:** EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
- **Interpretability:** Self-attention maps visualised per class

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
pip install scikit-learn tensorflow numpy matplotlib seaborn scipy vmdpy antropy imbalanced-learn scikit-optimize
```

### 4. Add the dataset
Download the Bonn dataset and place the folders (`Z`, `O`, `N`, `F`, `S`) inside a `dataset/` directory.

### 5. Run the notebooks in order
```
1. epilepsy_preprocessing.ipynb
2. epilepsy_cnn_model.ipynb
3. eeg_transformer_comparison.ipynb
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

| Model | Test Accuracy | Macro AUC | Balanced Accuracy | MCC | Notes |
|-------|:------------:|:---------:|:-----------------:|:---:|-------|
| 1D CNN | 95.04% | 0.9901 | 95.00% | 0.9268 | Hand-crafted features, Test Loss: 0.2247 |
| Bayesian SVM | **99.17%** | **0.9983** | **99.17%** | **0.9877** | Best CV AUC: 0.9964 |
| Transformer | 95.25% | 0.9933 | 95.31% | 0.9258 | Raw EEG windows, no feature engineering |

### Key observations

- The **SVM** remains the top performer on this dataset, benefiting from carefully engineered entropy and PCA features combined with Bayesian hyperparameter tuning.
- The **Transformer** achieves comparable accuracy to the CNN (95.25% vs 95.04%) while surpassing it on AUC (0.9933 vs 0.9901) — despite receiving no hand-crafted features, only raw waveforms.
- The Transformer's higher AUC indicates better-calibrated probability estimates, which is clinically valuable for seizure detection thresholding.
- The Transformer additionally produces **attention maps** that highlight which time steps within a window drive each prediction, offering a form of interpretability not available from the CNN.

### Best SVM Hyperparameters (Bayesian Optimization)

| Parameter | Value |
|-----------|-------|
| Kernel | RBF |
| C | 22.417 |
| gamma | 0.005966 |

### Model Comparison

| Aspect | 1D CNN | SVM (Bayesian) | Transformer |
|--------|--------|----------------|-------------|
| **Input** | Hand-crafted features | Hand-crafted features | Raw EEG windows |
| **Feature engineering** | Required | Required | Not needed |
| **Interpretability** | Low | Permutation importance | Attention maps |
| **Training speed** | Fast | Moderate | Moderate |
| **Scalability** | High | Limited | High |

---

## 👩‍💻 Author

**Shruthi Nagappan**
[GitHub](https://github.com/shruthi-nagappan/eeg-seizure-prediction)
