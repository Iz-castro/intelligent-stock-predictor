# core/data_preprocessing.py
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import re
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

    df = df.sort_values('Data')
    df = df.reset_index(drop=True)
    return df[['Data', 'Fechamento', 'Abertura', 'Maxima', 'Minima', 'Volume', 'Variacao']]


def preprocess_data(df, feature_col='Fechamento', sequence_length=60):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df[[feature_col]].values)

    X, y = [], []
    for i in range(sequence_length, len(scaled_data)):
        X.append(scaled_data[i-sequence_length:i, 0])
        y.append(scaled_data[i, 0])

    return scaler, pd.DataFrame(X), pd.DataFrame(y)


def split_data(X, y, test_size=0.2):
    return train_test_split(X, y, test_size=test_size, shuffle=False)


def merge_and_clean_csv(filepaths):
    all_dfs = [clean_and_format_dataframe(pd.read_csv(fp)) for fp in filepaths]
    merged_df = pd.concat(all_dfs).drop_duplicates(subset=['Data'])
    return merged_df.sort_values('Data').reset_index(drop=True)


def load_csv(filepath):
    return pd.read_csv(filepath, parse_dates=True, index_col=0)
