# 🧠 Air Quality Index Forecasting Using Hybrid Deep Learning

## 🔍 Introduction

Air pollution has become one of the most pressing environmental challenges of our time. Timely and accurate prediction of the Air Quality Index (AQI) is essential for environmental monitoring, public health awareness, and policy decision-making.

This project presents a hybrid deep learning framework combining a Time-Distributed CNN, Bidirectional LSTM, and Random Forest (CNN-BiLSTM-RF) for AQI prediction using hourly sensor data from Indian monitoring stations (2015–2020). The architecture uses CNN layers for spatial feature extraction, BiLSTM for long-range temporal dependency modelling, and Random Forest as a residual corrector on the extracted deep features.

The pipeline follows a strict chronological 70/15/15 train/validation/test split to ensure evaluation on genuinely future unseen data, with all preprocessing steps applied exclusively to the training fold to prevent data leakage.

---

## 📁 Dataset

The dataset (`station_hour.csv`) contains hourly air quality measurements from multiple Indian cities between 2015 and 2020, sourced from the Central Pollution Control Board (CPCB) via Kaggle.

**Features used:** PM2.5, PM10, NO, NO2, NOx, NH3, CO, SO2, O3, Benzene, Toluene, Xylene  
**Target:** AQI

Place the dataset in the project root directory before running.

---

## 🗂️ File Structure

```
├── main.py                  # Main training pipeline
├── main_analysis.py         # Loads saved models — runs baselines, ablation, residual analysis
├── data_preprocessing.py    # Data loading and outlier removal
├── feature_scaling.py       # PowerTransformer scaling (fit on train only)
├── cnn_lstm_model.py        # CNN-BiLSTM architecture, training, feature extraction
├── random_forest_model.py   # Random Forest training and hyperparameter tuning
├── evaluation.py            # Metrics computation on train/val/test sets
├── visualization.py         # All plots and charts
├── baselines.py             # SVR, RF, LSTM, GRU, CNN-GRU baselines on same dataset
├── ablation.py              # Component ablation study (CNN / BiLSTM / CNN+BiLSTM / full)
├── residual_analysis.py     # Residual diagnostics and error distribution analysis
├── requirements.txt         # Python dependencies
├── station_hour.csv         # Dataset (place locally)
└── LICENSE                  # Apache 2.0 License
```

---

## 🔄 Pipeline

### Data Pipeline (leakage-free)
1. Load raw data — nulls retained
2. Chronological 70/15/15 split on raw data — **no shuffling**
3. Null removal independently on each split
4. Outlier removal using Z-score (threshold = 5) — **train only**
5. PowerTransformer scaling — **fit on train, transform val and test**
6. Sequence construction (timesteps = 10)

### Model Pipeline
1. **CNN-BiLSTM** — extracts spatiotemporal features from sequences
2. **Feature extraction** — intermediate BiLSTM output used as RF input
3. **Random Forest** — trained on extracted features for final AQI prediction
4. **Evaluation** — all metrics reported on held-out test set only

---

## 📊 Results

### Proposed Model (Test Set)
| MAE | RMSE | R² | MSLE |
|---|---|---|---|
| 28.33 | 40.76 | 0.897 | 0.047 |

### Baseline Comparison (same dataset, same split, test set only)
| Model | MAE | RMSE | R² |
|---|---|---|---|
| SVR | 55.62 | 78.53 | 0.418 |
| Standalone RF | 48.88 | 67.92 | 0.565 |
| Unidirectional LSTM | 30.02 | 44.12 | 0.817 |
| GRU | 28.53 | 41.19 | 0.840 |
| CNN-GRU | 30.75 | 44.57 | 0.813 |
| **CNN-BiLSTM-RF (Proposed)** | **28.33** | **40.76** | **0.897** |

### Ablation Study (Test Set)
| Variant | MAE | RMSE | R² |
|---|---|---|---|
| A) CNN Only | 45.01 | 62.33 | 0.634 |
| B) BiLSTM Only | 27.99 | 41.07 | 0.841 |
| C) CNN + BiLSTM | 30.08 | 43.16 | 0.824 |
| **D) CNN + BiLSTM + RF (Proposed)** | **28.33** | **40.76** | **0.897** |

---

## 📉 Training Curves

Training and validation loss converge steadily with no signs of overfitting. Early stopping restores best weights automatically.

![Learning Curves](https://github.com/user-attachments/assets/12cb39cd-e4b7-4672-9fa1-b9205fcceee2)
---

## 📈 Prediction Visualisation

Monthly AQI accumulation chart (2015–2020):

![Figure 4 Month-wise plot of Average AQI](https://github.com/user-attachments/assets/11bbb98f-e3a1-490f-ab81-ece17f2f3568)

---
## 📈 Prediction Results

### Full Test Set — Actual vs Predicted AQI
![Actual vs Predicted AQI](plots/actual_vs_predicted.png)

The predicted values (orange) closely track the actual AQI (blue) across the full held-out test set (~9,200 samples). The model captures seasonal trends and general pollution dynamics well, with expected underestimation at extreme peak values.

## 🚀 Running the Project

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Place dataset
Download `station_hour.csv` and place it in the project root.

### 3. Train the proposed model
```bash
python main.py
```
This saves `cnn_lstm_model.keras`, `best_rf_model.pkl`, and `tgt_scaler.pkl`.

### 4. Run baselines, ablation, and residual analysis
```bash
python main_analysis.py
```
This loads the saved models — no retraining required.

---

## 📜 License

This project is licensed under the Apache 2.0 License. You are free to use, modify, and distribute the code with proper attribution.
