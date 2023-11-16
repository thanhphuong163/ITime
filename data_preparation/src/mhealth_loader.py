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
import ipywidgets as widgets
# import bqplot.pyplot as bqplt
# from tqdm.notebook import tqdm
from IPython.core.interactiveshell import InteractiveShell
import warnings
from pandas.core.common import SettingWithCopyWarning
warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)

features = ['acc_chest_x', 'acc_chest_y', 'acc_chest_z', 'ecd_chest_1', 'ecd_chest_2',
            'acc_lankle_x', 'acc_lankle_y', 'acc_lankle_z', 'gyro_lankle_x', 'gyro_lankle_y', 'gyro_lankle_z', 'mag_lankle_x', 'mag_lankle_y', 'mag_lankle_z',
            'acc_rlarm_x', 'acc_rlarm_y', 'acc_rlarm_z', 'gyro_rlarm_x', 'gyro_rlarm_y', 'gyro_rlarm_z', 'mag_rlarm_x', 'mag_rlarm_y', 'mag_rlarm_z',
            'label']


class MHealthDataset:
    def __init__(self, data_path, activities=None, train_rate=0.5, nb_views=5):
        self.data_path = data_path
        self.nb_views = nb_views
        self.features = features
        self.train_rate = train_rate
        self.activities = activities

    def load_data(self):
        p_paths = sorted(glob.glob(self.data_path+"/*.log"))
        p_dfs = [pd.read_csv(
            p_path, delimiter='\t', names=self.features, header=None) for p_path in p_paths]
        aps_dfs = {}
        min_length = 100000
        for p, p_df in enumerate(p_dfs):
            tmp_dfs = [group[1] for group in p_df.groupby(
                'label') if group[0] in self.activities]
            for act in self.activities:
                key = f"a{act:02d}_p{p+1}"
                aps_dfs[key] = tmp_dfs[act-1]
                min_length = min(min_length, aps_dfs[key].shape[0])
        train_ap_dfs = {}
        test_ap_dfs = {}
        # train_split = int(min_length*self.train_rate)
        train_split = 1048
        for key, ap_df in aps_dfs.items():
            train_ap_dfs[key] = aps_dfs[key][:train_split].reset_index().drop(columns=[
                'index'])
            test_ap_dfs[key] = aps_dfs[key][train_split:train_split+train_split].reset_index().drop(columns=[
                'index'])
        return train_ap_dfs, test_ap_dfs

    def split_views(self, ap_dfs):
        views = {}
        for v, part in enumerate(['chest', 'lankle', 'rlarm']):
            view_features = [feat for feat in self.features if part in feat]
            views[f'view_{v+1}'] = {key: df[view_features]
                                    for key, df in ap_dfs.items()}
        return views
