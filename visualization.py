import matplotlib.pyplot as plt
import seaborn as sns

def plot_actual_vs_predicted(y_test_inv, y_test_pred_inv):
    """
    Plot the actual vs predicted AQI values for the test set.
    :param y_test_inv: Actual AQI values for the test set (inverse transformed).
    :param y_test_pred_inv: Predicted AQI values for the test set (inverse transformed).
    :return: None
    """
    plt.figure(figsize=(10, 5))
    plt.plot(y_test_inv, label='Actual AQI', color='blue')
    plt.plot(y_test_pred_inv, label='Predicted AQI', color='orange', alpha=0.7)
    plt.legend()
    plt.title('Actual vs Predicted AQI (Test Set)')
    plt.xlabel('Samples')
    plt.ylabel('AQI')
    plt.grid(True)
    plt.show(block=False)  # Display plot without blocking
      

def plot_learning_curves(history):
    """
    Plot the learning curves for training and validation loss.
    :param history: The history object from model fitting (contains loss values).
    :return: None
    """
    plt.figure(figsize=(12, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Learning Curves')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show(block=False)  # Display plot without blocking
     

def plot_distribution_before_scaling(df, features_to_scale, colors):
    """
    Plot distribution of features before scaling/transformation.
    :param df: DataFrame containing the original features.
    :param features_to_scale: List of features to plot.
    :param colors: List of colors for each plot.
    """
    for i, col in enumerate(features_to_scale):
        plt.figure(figsize=(7, 5))
        sns.histplot(df[col], kde=True, color=colors[i % len(colors)])
        plt.title(f'Before Scaling/Transformation: {col}', fontsize=14, fontweight='bold', color='black')
        plt.xlabel(col, fontsize=12, fontweight='bold', color='black')
        plt.ylabel('Frequency', fontsize=12, fontweight='bold', color='black')
        plt.grid(True)
        plt.show(block=False)  # Display plot without blocking
        plt.pause(1)  # Pause for 1 second before moving to the next plot

def plot_boxplot_before_scaling(df, features_to_scale, colors):
    """
    Plot box plot of features before scaling/transformation.
    :param df: DataFrame containing the original features.
    :param features_to_scale: List of features to plot.
    :param colors: List of colors for each plot.
    """
    for i, col in enumerate(features_to_scale):
        plt.figure(figsize=(7, 5))
        sns.boxplot(y=df[col], color=colors[i % len(colors)])
        plt.title(f'Box Plot Before Scaling/Transformation: {col}', fontsize=14, fontweight='bold', color='black')
        plt.xlabel(col, fontsize=12, fontweight='bold', color='black')
        plt.grid(True)
        plt.show(block=False)  # Display plot without blocking
        plt.pause(1)  # Pause for 1 second before moving to the next plot

def plot_distribution_after_scaling(df_scaled, features_to_scale, colors):
    """
    Plot distribution of features after scaling/transformation.
    :param df_scaled: DataFrame containing the scaled features.
    :param features_to_scale: List of features to plot.
    :param colors: List of colors for each plot.
    """
    for i, col in enumerate(features_to_scale):
        plt.figure(figsize=(7, 5))
        sns.histplot(df_scaled[col], kde=True, color=colors[i % len(colors)])
        plt.title(f'After Scaling/Transformation: {col}', fontsize=14, fontweight='bold', color='black')
        plt.xlabel(col, fontsize=12, fontweight='bold', color='black')
        plt.ylabel('Frequency', fontsize=12, fontweight='bold', color='black')
        plt.grid(True)
        plt.show(block=False)  # Display plot without blocking
        plt.pause(1)  # Pause for 1 second before moving to the next plot

def plot_boxplot_after_scaling(df_scaled, features_to_scale, colors):
    """
    Plot box plot of features after scaling/transformation.
    :param df_scaled: DataFrame containing the scaled features.
    :param features_to_scale: List of features to plot.
    :param colors: List of colors for each plot.
    """
    for i, col in enumerate(features_to_scale):
        plt.figure(figsize=(7, 5))
        sns.boxplot(y=df_scaled[col], color=colors[i % len(colors)])
        plt.title(f'Box Plot After Scaling/Transformation: {col}', fontsize=14, fontweight='bold', color='black')
        plt.xlabel(col, fontsize=12, fontweight='bold', color='black')
        plt.grid(True)
        plt.show(block=False)  # Display plot without blocking
        plt.pause(1)  # Pause for 1 second before moving to the next plot
