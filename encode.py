import pandas as pd
import numpy as np

'''
# ARC SIN TRANSFORM
def encode_arcsin_transform(df: pd.DataFrame) -> pd.DataFrame:
    scale_factor = 72.0
    max_value = 1e6  # match data's max intensity
    df["int"] = df["int"].apply(
        lambda x: (np.floor(np.arcsin(np.sqrt(x / max_value)) * scale_factor)).astype(np.uint8)
    )
    return df

# YEO JOHNSON TRANSFORM
def encode_yeojohnson_transform(df: pd.DataFrame) -> pd.DataFrame:
    scale_factor = 72.0
    lmbda = 0.5  # based on data distribution
    df["int"] = df["int"].apply(
        lambda x: (np.floor((( (x + 1) ** lmbda - 1 ) / lmbda ) * scale_factor )).astype(np.uint8)
    )
    return df
# BOX COX TRANSFORM (+)
def encode_boxcox_transform(df: pd.DataFrame) -> pd.DataFrame:
    scale_factor = 72.0
    lmbda = 0.5  # based on data distribution
    # add 1 to handle x=0 (strictly positive for Box-Cox)
    df["int"] = df["int"].apply(
        lambda x: (np.floor(( ((x + 1) ** lmbda - 1 ) / lmbda ) * scale_factor )).astype(np.uint8)
    )
    return df

# SQUARE ROOT TRANSFORM (+)
def encode_sqrt_transform(df: pd.DataFrame) -> pd.DataFrame:
    scale_factor = 72.0
    df["int"] = df["int"].apply(
        lambda x: (np.floor(np.sqrt(x) * scale_factor)).astype(np.uint8)
    )
    return df

# ORIGINAL (+)
def encode_log_transform(df: pd.DataFrame) -> pd.DataFrame:
    # An example encoding function is provided:
    # Take the log of the intensities ('int' column), add 1 to avoid log(0), multiply by a scale factor, floor and store as uint8
    scale_factor = 72.0
    df["int"] = df["int"].apply(
        lambda x: (np.floor(np.log2(x + 1) * scale_factor)).astype(np.uint8)
    )
    return df

def encoder(df: pd.DataFrame) -> pd.DataFrame:
    # Call your encoding function here and return transformed DataFrame
    encoded_df = encode_boxcox_transform(df)
    return encoded_df
'''
import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull


def encode_top(df: pd.DataFrame) -> pd.DataFrame:
    def apply_mask(row):
        if len(row['mz']) < 60:
            row['mz'] = [0]
            row['int'] = [0]
        else:
            points = np.column_stack((row['mz'], row['int']))
            hull = ConvexHull(points)
            hull_indices = np.unique(hull.vertices)  # Get unique convex hull vertices

            row['mz'] = row['mz'][hull_indices].astype(np.float32)
            row['int'] = row['int'][hull_indices].astype(np.float32)
        return row

    # Apply the mask to each row
    df = df.apply(apply_mask, axis=1)
    df.drop(columns=['retention_time', 'ms_level', 'int'], inplace=True)
    return df


def encoder(df: pd.DataFrame) -> pd.DataFrame:
    # Call your encoding function here and return transformed DataFrame
    encoded_df = encode_top(df)['mz'].to_numpy()
    return encoded_df
