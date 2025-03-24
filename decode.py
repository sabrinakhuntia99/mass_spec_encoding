import pandas as pd
import numpy as np

'''
# ARC SIN TRANSFORM

def decode_arcsin_transform(df: pd.DataFrame) -> pd.DataFrame:
    scale_factor = 72.0
    max_value = 1e6
    df["int"] = df["int"].apply(
        lambda y: (np.sin(y / scale_factor) ** 2 * max_value
    ).astype(np.float64))
    return df


# YEO JOHNSON TRANSFORM
def decode_yeojohnson_transform(df: pd.DataFrame) -> pd.DataFrame:
    scale_factor = 72.0
    lmbda = 0.5
    df["int"] = df["int"].apply(
        lambda y: (( (y / scale_factor ) * lmbda + 1 ) ** (1 / lmbda) ) - 1
    ).astype(np.float64)
    return df

# BOX COX TRANSFORM
def decode_boxcox_transform(df: pd.DataFrame) -> pd.DataFrame:
    scale_factor = 72.0
    lmbda = 0.5
    df["int"] = df["int"].apply(
        lambda y: ((( (y / scale_factor ) * lmbda + 1 ) ** (1 / lmbda) ) - 1).astype(np.float64)
    )
    return df


# SQUARE ROOT TRANSFORM

def decode_sqrt_transform(df: pd.DataFrame) -> pd.DataFrame:
    scale_factor = 72.0
    df["int"] = df["int"].apply(
        lambda y: ((y / scale_factor) ** 2).astype(np.float64)
    )
    return df


def decode_log_transform(df: pd.DataFrame) -> pd.DataFrame:
    # An example encoding function is provided:
    # Apply the exponential to reverse the log transform, divide by scale factor to reverse scaling, then subtract 1
    scale_factor = 72.0
    df["int"] = df["int"].apply(
        lambda x: (np.exp2(x / scale_factor) - 1).astype(np.float64)
    )
    return df



def decoder(df: pd.DataFrame) -> pd.DataFrame:
    # Call your decoding function here and return transformed DataFrame
    decoded_df = decode_boxcox_transform(df)
    return decoded_df
'''
def decode_top(df: pd.DataFrame) -> pd.DataFrame:
    # for each row, add a list of 1s of the same length as the 'mz' column
    df['int'] = df['mz'].apply(lambda x: [1.0] * len(x))
    df['ms_level'] = 2
    df['retention_time'] = list(range(len(df)))
    df['mz'] = df['mz'].apply(lambda x: np.array(x, dtype=np.float64).tolist())

    return df


def decoder(df: pd.DataFrame) -> pd.DataFrame:
    # Call your decoding function here and return transformed DataFrame
    decoded_df = decode_top(df)
    return decoded_df
