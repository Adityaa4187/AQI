from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Flatten, LSTM, Dense, TimeDistributed, Dropout, Bidirectional, BatchNormalization
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.models import Sequential

def build_cnn_lstm_model(input_shape):
    """
    Build a CNN-LSTM model for time series forecasting.
    :param input_shape: Shape of the input data (timesteps, features).
    :return: Compiled CNN-LSTM model.
    """
    cnn_lstm_model = Sequential([
        TimeDistributed(Conv1D(filters=128, kernel_size=3, activation='relu'), input_shape=(None, input_shape[1], 1)),
        TimeDistributed(Conv1D(filters=64, kernel_size=3, activation='relu')),
        TimeDistributed(MaxPooling1D(pool_size=2)),
        TimeDistributed(BatchNormalization()),
        TimeDistributed(Flatten()), 
        Bidirectional(LSTM(100, activation='tanh', return_sequences=False)),
        Dropout(0.3),
        Dense(1, kernel_regularizer='l2')
    ])
    cnn_lstm_model.compile(optimizer='adam', loss='mse')
    return cnn_lstm_model

def train_cnn_lstm_model(cnn_lstm_model, X_train, y_train, X_val, y_val):
    """
    Train the CNN-LSTM model.
    :param cnn_lstm_model: Compiled CNN-LSTM model.
    :param X_train: Training feature data.
    :param y_train: Training target data.
    :param X_val: Validation feature data.
    :param y_val: Validation target data.
    :return: Training history.
    """
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)
    history = cnn_lstm_model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_val, y_val),
                                 callbacks=[early_stopping, reduce_lr])
    return history

def extract_features(cnn_lstm_model, X):
    """
    Extract features using the CNN-LSTM model.
    :param cnn_lstm_model: Trained CNN-LSTM model.
    :param X: Input data to extract features from.
    :return: Extracted features.
    """
    feature_extractor = Sequential(cnn_lstm_model.layers[:-2])
    return feature_extractor.predict(X)
