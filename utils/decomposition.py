from statsmodels.tsa.seasonal import STL

def decompose(df):
    stl = STL(df["energy"], period=24)
    res = stl.fit()

    df["trend"] = res.trend
    df["seasonal"] = res.seasonal
    df["residual"] = res.resid
    
    return df