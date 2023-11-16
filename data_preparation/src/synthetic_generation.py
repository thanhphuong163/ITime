from anomaly_generations import swap_time_steps
from dsa_loader import DSADataset
from mex_loader import MEXDataset
from mhealth_loader import MHealthDataset
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


parser = argparse.ArgumentParser(description='Enter some arguments.')
parser.add_argument('--dataset_name', type=str, help='enter a dataset name.')
parser.add_argument('--n_samples', type=int, help='enter sample.')
parser.add_argument('--n_views', type=int, help='enter number of views')
parser.add_argument('--n_clusters', type=int, help='enter number of clusters')
args = parser.parse_args()


if __name__ == "__main__":
    print("Loading dataset...")
    
    if 'dsa' in args.dataset_name:
        clusters = [f"a{i:02d}" for i in range(1, args.n_clusters+1)]
        dataset = DSADataset("../raw_datasets/dsa", activities=clusters)
        train_object_dfs, test_object_dfs = dataset.load_data()
        train_views_dfs = dataset.split_views(train_object_dfs)
        test_views_dfs = dataset.split_views(test_object_dfs)
    elif 'mex' in args.dataset_name:
        exercises = ['01', '02', '03', '05', '06', '07']
        clusters = [f'e{ex}' for ex in exercises]
        mex_dataset = MEXDataset("../raw_datasets/mex", exercises=exercises)
        train_views_dfs, test_views_dfs = mex_dataset.load_data()
    elif 'mhealth' in args.dataset_name:
        activities = list(range(1, args.n_clusters+1))
        clusters = [f"a{act:02d}" for act in activities]
        dataset = MHealthDataset('../raw_datasets/mhealth', activities=activities)
        train_ap_dfs, test_ap_dfs = dataset.load_data()
        train_views_dfs = dataset.split_views(train_ap_dfs)
        test_views_dfs = dataset.split_views(test_ap_dfs)
    else:
        raise ValueError(f"This dataset {args.dataset_name} is not available.")
    

    for sample in range(1, args.n_samples+1):
        print(f"Generating sample {sample}...")
        for anomaly_rate in [5, 10, 15, 20]:
            print(f"Generating anomaly rate {anomaly_rate}...")
            dir_path = f"../preprocessed_datasets/{args.dataset_name}/sample{sample}/anomaly_rate_{anomaly_rate}_views_{args.n_views}"
            # Generate anomalies
            swapped_test_views_dfs, ground_truths = swap_time_steps(
                copy.deepcopy(test_views_dfs),
                clusters=copy.deepcopy(clusters),
                anomaly_rate=anomaly_rate*0.01)

            # Save to files
            print("Saving files...")
            for view, view_dfs in train_views_dfs.items():
                view_path = dir_path+f"/train/{view}"
                if not os.path.exists(view_path):
                    os.makedirs(view_path)
                for ap, df in view_dfs.items():
                    df.to_csv(f"{view_path}/{ap}.csv", index=False)
            for view, view_dfs in swapped_test_views_dfs.items():
                view_path = dir_path+f"/test/{view}"
                if not os.path.exists(view_path):
                    os.makedirs(view_path)
                for ap, df in view_dfs.items():
                    df.to_csv(f"{view_path}/{ap}.csv", index=False)
            for ap, gt in ground_truths.items():
                gt.to_csv(dir_path+f"/test/{ap}.csv", index=False)
            print('Done.')
