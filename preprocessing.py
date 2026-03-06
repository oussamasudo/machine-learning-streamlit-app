import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

def handle_missing_values(df, method="moyenne"):
    if method == "supprimer les lignes":
        return df.dropna()
    
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            if method == "valeur fixe":
                if df[col].dtype == "object":
                    df[col] = df[col].fillna("Missing")
                else:
                    df[col] = df[col].fillna(0)
            elif method == "médiane":
                if df[col].dtype == "object":
                    df[col] = df[col].fillna(df[col].mode()[0])
                else:
                    df[col] = df[col].fillna(df[col].median())
            else: # moyenne or default
                if df[col].dtype == "object":
                    df[col] = df[col].fillna(df[col].mode()[0])
                else:
                    df[col] = df[col].fillna(df[col].mean())
    return df

def encode_categorical(df):
    encoders = {}
    for col in df.select_dtypes(include="object").columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le
    return df, encoders

def normalize_features(df, method='StandardScaler', exclude_target=True):
    """
    Normalise uniquement les features, PAS la variable cible
    """
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    
    # Exclure la dernière colonne si c'est le target
    if exclude_target and len(numeric_cols) > 0:
        # Supposant que le target est la dernière colonne ou appelé 'target'
        if 'target' in numeric_cols:
            numeric_cols.remove('target')
    
    if len(numeric_cols) == 0:
        return df
    
    if method == 'StandardScaler':
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
    elif method == 'MinMaxScaler':
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
    else:
        from sklearn.preprocessing import RobustScaler
        scaler = RobustScaler()
    
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    return df
