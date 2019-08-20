import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


class DS:
    """Class to encapsulate names and paths and fields for a dataset"""
    
    def __init__(self, ds_name, target_field):
        self.ds_name = ds_name
        self.target_field = target_field

        def get_ds(self):
            pass

ix = DS('ix','Music')

def get_ix_ds(drop_infrequent_labels=False):
    ds_csv_file_name = 'cleaned_index_data.csv'
    df = pd.read_csv(ds_csv_file_name)
    df.set_index('date', inplace=True)
    if drop_infrequent_labels:
        low_labels = df[ix.target_field].value_counts() <= 2
        drop_list = list(low_labels[low_labels].index)
        df = df[~df['Music'].isin(drop_list)]
    X = df.drop(ix.target_field, 1).copy().values
    Y_categorical = df[ix.target_field].copy().values
    le = LabelEncoder()
    le.fit(Y_categorical)
    Y = le.transform(Y_categorical)
    # Scikit learn really wants floats or the scaler will complain
    X = X.astype(np.float64)
    return X, Y

ix.get_ds = get_ix_ds