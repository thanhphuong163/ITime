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

features = [
    'T_xacc', 'T_yacc', 'T_zacc', 'T_xgyro', 'T_ygyro', 'T_zgyro', 'T_xmag', 'T_ymag', 'T_zmag',
    'RA_xacc', 'RA_yacc', 'RA_zacc', 'RA_xgyro', 'RA_ygyro', 'RA_zgyro', 'RA_xmag', 'RA_ymag', 'RA_zmag',
    'LA_xacc', 'LA_yacc', 'LA_zacc', 'LA_xgyro', 'LA_ygyro', 'LA_zgyro', 'LA_xmag', 'LA_ymag', 'LA_zmag',
    'RL_xacc', 'RL_yacc', 'RL_zacc', 'RL_xgyro', 'RL_ygyro', 'RL_zgyro', 'RL_xmag', 'RL_ymag', 'RL_zmag',
    'LL_xacc', 'LL_yacc', 'LL_zacc', 'LL_xgyro', 'LL_ygyro', 'LL_zgyro', 'LL_xmag', 'LL_ymag', 'LL_zmag']


class DSADataset:
    def __init__(self, data_path, activities=None, train_rate=0.5, nb_views=5):
        self.data_path = data_path
        self.nb_views = nb_views
        self.features = features
        self.train_rate = train_rate
        self.activities = activities

    def load_data(self):
        a_paths = []
        if self.activities is None:
            a_paths = sorted(glob.glob(f"{self.data_path}/*"))
        else:
            a_paths = []
            for a in self.activities:
                a_paths += sorted(glob.glob(f"{self.data_path}/{a}"))
        train_ap_dfs = {}
        test_ap_dfs = {}
        for a_path in a_paths:
            print(f"Loading {a_path}...")
            a_key = a_path.split('/')[-1]
            p_paths = sorted(glob.glob(f"{a_path}/*"))
            for p_path in p_paths:
                p_key = p_path.split('/')[-1]
                s_paths = sorted(glob.glob(f"{p_path}/*.txt"))
                s_dfs = [
                    pd.read_csv(s_path, header=None, names=self.features)
                    for s_path in s_paths
                    ]
                p_df = pd.concat(s_dfs, axis=0).reset_index()\
                                               .drop(columns=['index'])
                train_split = int(p_df.shape[0] * self.train_rate)
                # scaler = StandardScaler()
                # train_data = scaler.fit_transform(p_df[:train_split])
                # train_ap_dfs[f"{a_key}_{p_key}"] = pd.DataFrame(data=train_data, columns=p_df.columns)
                # test_data = scaler.transform(p_df[train_split:])
                # test_ap_dfs[f"{a_key}_{p_key}"] = pd.DataFrame(data=test_data, columns=p_df.columns)
                train_ap_dfs[f"{a_key}_{p_key}"] = p_df#[:train_split]
                test_ap_dfs[f"{a_key}_{p_key}"] = p_df#[train_split:].reset_index().drop(columns=['index'])
        return train_ap_dfs, test_ap_dfs

    def split_views(self, ap_dfs):
        views = {}
        for v, part in enumerate(['T', 'RA', 'LA', 'RL', 'LL']):
            view_features = [feat for feat in self.features if part in feat]
            views[f'view_{v+1}'] = {key: df[view_features]
                                    for key, df in ap_dfs.items()}
        return views
    
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
