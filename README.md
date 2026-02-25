# 🧠 Project Title: Air Quality Index Forecasting Using Hybrid Deep Learning Models
## 🔍 Introduction

Air pollution has become one of the most pressing environmental challenges of our time. Timely and accurate prediction of the Air Quality Index (AQI) is essential for environmental monitoring, public health awareness, and decision-making. This project presents a hybrid deep learning framework that combines Time Distributed CNN, Bidirectional LSTM, and Random Forest for accurate AQI prediction across Indian metropolitan regions from 2015 to 2020.

The hybrid model leverages both spatial and temporal patterns in pollution data. The CNN component is used for efficient feature extraction, while the BiLSTM learns long-range temporal dependencies. A Random Forest layer on top enhances predictive robustness.

## 📁 Dataset:

The dataset (station_hour.csv) is available in the repo and must be downloaded locally before execution. It includes hourly air quality data from multiple Indian cities between 2015 and 2020.

## 🗂️ Project File Structure
    ├── main.py                 # Main script for training and evaluation
    ├── data_preprocessing.py  # Data loading & outlier removal
    ├── feature_scaling.py     # Feature scaling module
    ├── cnn_lstm_model.py      # CNN-LSTM architecture, training, feature extraction
    ├── random_forest_model.py # Random Forest training & tuning
    ├── evaluation.py          # Evaluation metrics & predictions
    ├── visualization.py       # Plots and charts
    ├── requirements.txt       # Python dependencies
    ├── station_hour.csv       # Dataset (place locally after downloading from Drive)
    └── LICENSE                # Apache 2.0 License


## 🔄 Workflow Summary

    1. Data Cleaning
        Outlier removal using Z-score transformation
        Handling missing values
    2. Feature Scaling
        Using PowerTransformer for normalization
    3. Sequence Preparation
        Reshaping data for LSTM using time windowing
    4. CNN-LSTM Model
        Extracts spatiotemporal patterns from input features
    5. Feature Extraction
        Intermediate CNN-LSTM outputs used as RF input
    6. Random Forest Model
        Trained on extracted deep features for final AQI prediction
    7. Evaluation
        Plots, prediction-vs-actual comparison, and error metrics


## 📊 Exploratory Data Analysis

Here’s a monthly AQI accumulation chart from 2015 to 2020, which helps identify monthly pollution patterns:

![Figure 4 Month-wise plot of Average AQI](https://github.com/user-attachments/assets/11bbb98f-e3a1-490f-ab81-ece17f2f3568)


📉 Model Training Results

The model shows excellent convergence, with training and validation losses decreasing steadily, indicating strong generalization.

![image](https://github.com/user-attachments/assets/acc74529-117c-49f8-bc40-b620d2b01206)


## 🚀 Running the Project

1. Install dependencies

        pip install -r requirements.txt

2. Download the dataset and place it in the project root directory as station_hour.csv.

3. Run the main script

       python main.py

## 📜 License

This project is licensed under the Apache 2.0 License.

You're free to use, modify, and distribute the code with proper attribution.

