import pandas as pd
import numpy as np
import os

def load_data():
    data_path = os.path.join('data', 'raw', 'bank-additional-full.csv')
    df = pd.read_csv(data_path, sep=';')
    return df

