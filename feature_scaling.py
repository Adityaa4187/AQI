from sklearn.preprocessing import PowerTransformer

def scale_features(df, features_to_scale, target_feature):
    """
    Apply PowerTransformer to scale features and target.
    :param df: The input DataFrame.
    :param features_to_scale: List of feature columns to scale.
    :param target_feature: The target column to scale.
    :return: Scaled DataFrame.
    """
    power_transformer = PowerTransformer(method='yeo-johnson', standardize=True)
    df_scaled = df.copy()
    df_scaled[features_to_scale] = power_transformer.fit_transform(df[features_to_scale])
    df_scaled[target_feature] = power_transformer.fit_transform(df[[target_feature]])
    return df_scaled, power_transformer
