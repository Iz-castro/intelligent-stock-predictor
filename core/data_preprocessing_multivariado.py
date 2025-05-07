# core/data_preprocessing_02.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from datetime import datetime

def parse_volume(volume_str):
    if isinstance(volume_str, str):
        volume_str = volume_str.replace('.', '').replace(',', '.')
        if 'K' in volume_str:
            return float(volume_str.replace('K', '')) * 1e3
        elif 'M' in volume_str:
            return float(volume_str.replace('M', '')) * 1e6
    return float(volume_str)

def convert_date(date_str):
    try:
        return datetime.strptime(date_str, "%d.%m.%Y").strftime("%Y-%m-%d")
    except:
        return date_str

def convert_brasilian_format(x):
    if isinstance(x, str):
        return float(x.replace('.', '').replace(',', '.'))
    return x

def compute_features(df):
    df['Retorno_%'] = df['Fechamento'].pct_change() * 100
    df['MM9'] = df['Fechamento'].rolling(window=9).mean()
    delta = df['Fechamento'].diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14).mean()
    avg_loss = pd.Series(loss).rolling(window=14).mean()
    rs = avg_gain / (avg_loss + 1e-6)
    df['RSI'] = 100 - (100 / (1 + rs))
    return df

def clean_and_format_dataframe(df):
    column_map = {
        'Último': 'Fechamento',
        'Máxima': 'Maxima',
        'Mínima': 'Minima',
        'Vol.': 'Volume',
        'Var%': 'Variacao'
    }
    df = df.rename(columns=column_map)
    df['Data'] = df['Data'].apply(convert_date)

    for col in ['Fechamento', 'Abertura', 'Maxima', 'Minima']:
        df[col] = df[col].apply(convert_brasilian_format)

    df['Volume'] = df['Volume'].apply(parse_volume)
    df['Variacao'] = df['Variacao'].str.replace('%', '').str.replace(',', '.')
    df['Variacao'] = pd.to_numeric(df['Variacao'], errors='coerce')

    df = df.sort_values('Data').reset_index(drop=True)
    df = compute_features(df)
    df = df.dropna()
    return df[['Data', 'Fechamento', 'Abertura', 'Maxima', 'Minima', 'Volume', 'Variacao', 'Retorno_%', 'MM9', 'RSI']]

def preprocess_data(df, feature_cols=['Fechamento', 'Retorno_%', 'MM9', 'RSI'], sequence_length=60):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df[feature_cols].values)

    X, y = [], []
    for i in range(sequence_length, len(scaled_data)):
        X.append(scaled_data[i-sequence_length:i])
        y.append(scaled_data[i, 0])  # prever fechamento

    return scaler, np.array(X), pd.DataFrame(y)

def split_data(X, y, test_size=0.2):
    return train_test_split(X, y, test_size=test_size, shuffle=False)

def merge_and_clean_csv(filepaths):
    all_dfs = [clean_and_format_dataframe(pd.read_csv(fp)) for fp in filepaths]
    merged_df = pd.concat(all_dfs).drop_duplicates(subset=['Data'])
    return merged_df.sort_values('Data').reset_index(drop=True)

def load_csv(filepath):
    return pd.read_csv(filepath, parse_dates=True, index_col=0)
