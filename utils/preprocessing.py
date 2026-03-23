def create_features(df):
    df["hour"] = df.index.hour
    df["dayofweek"] = df.index.dayofweek
    
    df["lag1"] = df["energy"].shift(1)
    df["lag2"] = df["energy"].shift(2)
    
    df.dropna(inplace=True)
    return df