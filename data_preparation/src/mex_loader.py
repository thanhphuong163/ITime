import argparse
import os
import sys
import glob
import copy
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import datetime
from datetime import datetime
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import roc_auc_score
# import ipywidgets as widgets
# import bqplot.pyplot as bqplt
# from tqdm.notebook import tqdm
from IPython.core.interactiveshell import InteractiveShell
import warnings
from pandas.core.common import SettingWithCopyWarning
warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)

features = ['act1', 'act2', 'act3', 'acw1', 'acw2', 'acw3']



class MEXDataset:
    def __init__(self, data_path, exercises=None, train_rate=0.5, nb_views=2):
        self.data_path = data_path
        self.nb_views = nb_views
        self.features = features
        self.train_rate = train_rate
        self.exercises = exercises
        self.views = ['act', 'acw']

    def load_data(self):
        train_views_dfs, test_views_dfs = {}, {}
        min_length = 10000000
        tmp_views_dfs = {}
        for v, view in enumerate(self.views):
            print(f'Loading {view}...')
            p_paths = sorted(glob.glob(f"{self.data_path}/{view}/*"))
            features = ['time'] + \
                [feat for feat in self.features if view in feat]
            tmp_views_dfs[f'view_{v+1}'] = {}
            for p_path in p_paths:
                p_key = p_path.split('/')[-1]
                e_paths = [
                    f"{p_path}/{ex}_{view}_1.csv" for ex in self.exercises]
                dfs = {
                    f"e{ex}_p{p_key}": pd.read_csv(path, names=features).drop(columns=['time'])
                    for ex, path in zip(self.exercises, e_paths)
                }
                min_length = min(min_length, *[df.shape[0]
                                 for df in dfs.values()])
                for key, df in dfs.items():
                    tmp_views_dfs[f'view_{v+1}'][key] = df
        train_split = int(min_length*self.train_rate)
        for view, view_dfs in tmp_views_dfs.items():
            train_views_dfs[view], test_views_dfs[view] = {}, {}
            for ep, df in view_dfs.items():
                train_views_dfs[view][ep] = df#[:train_split]
                test_views_dfs[view][ep] = df[train_split:min_length].reset_index().drop(columns=[
                    'index'])
        return train_views_dfs, test_views_dfs
    
    def save_into_features(self, stored_dir, train_views_dfs):
        print("Saving features into files")
        for view, view_dfs in train_views_dfs.items():
            view_path = stored_dir+f"/raw/{view}"
            if not os.path.exists(view_path):
                os.makedirs(view_path)
            for ap, df in view_dfs.items():
                for col in df.columns:
                    path = f"{view_path}/{col}"
                    os.makedirs(path, exist_ok=True)
                    df[col].to_csv(f"{path}/{ap}.csv", header=[col])
