def detect_anomalies(df, threshold=2):
    mean = df["Amount"].mean()
    std = df["Amount"].std()
    df["z_score"] = (df["Amount"] - mean) / std
    return df[df["z_score"].abs() > threshold]