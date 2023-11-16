import glob
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler, MinMaxScaler

class MyDataset(Dataset):
    def __init__(self, data_dir, stage, window_size):
        super(MyDataset, self).__init__()
        self.data_dir = data_dir
        self.window_size = window_size
        self.stage = stage
        if stage == 'fit':
            self.data = self.load_data(f"{data_dir}/train")
        elif stage == 'test':
            self.data = self.load_data(f"{data_dir}/test")
            self.ground_truth = self.load_ground_truth(f"{data_dir}/test")
        elif stage == 'predict':
            self.data = self.load_data(f"{data_dir}/test")
        self.view_szes = [dfs[0].shape[1] for dfs in self.data.values()]

    @staticmethod
    def normalize_dataset(df):
        # scaler = MinMaxScaler()
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(df)
        return scaled_data

    def load_data(self, data_dir):
        view_paths = sorted(glob.glob(data_dir+"/view*"))
        views_dfs = {view_path[-6:]: None for view_path in view_paths}
        for view_path in view_paths:
            print(view_path)
            p_paths = sorted(glob.glob(view_path+"/*.csv"))
            view_dfs = []
            for path in p_paths:
                df = pd.read_csv(path, dtype=np.float32)
                data = self.normalize_dataset(df)
                view_dfs.append(data)
            views_dfs[view_path[-6:]] = view_dfs
        # Get length of all mts
        self.lens = [df.shape[0] - self.window_size for df in views_dfs['view_1']]
        return views_dfs

    def load_ground_truth(self, data_dir):
        paths = sorted(glob.glob(data_dir+"/*.csv"))
        dfs = [pd.read_csv(path, index_col=False, dtype=np.int32)['is_anomaly'] for path in paths]
        return dfs

    def __len__(self):
        return sum(self.lens)

    def __getitem__(self, index):
        i = 0
        for len_ in self.lens:
            if index >= len_:
                index -= len_
                i += 1
            else:
                break
        X = [torch.tensor(view_dfs[i][index:index+self.window_size]) for v, view_dfs in self.data.items()]
        # if self.stage in ['fit', 'predict']:
        #     y = torch.tensor(0)
        # else:
            # y = torch.tensor(self.ground_truth[i][index+self.window_size-1])
        y = torch.tensor(self.ground_truth[i][index+self.window_size-1]) if self.stage == 'test' else torch.tensor(0)
        # X_next = [torch.tensor(view_dfs[i][index+self.window_size:index+self.window_size+1]) for v, view_dfs in self.data.items()]
        return X, y#, X_next


# class FullSequenceDataset(Dataset):
#     def __init__(self, data_dir, stage, window_size):
#         super(FullSequenceDataset, self).__init__()
#         self.data_dir = data_dir
#         self.window_size = window_size
#         self.stage = stage
#         if stage == 'fit':
#             self.data = self.load_data(f"{data_dir}/train")
#         elif stage == 'test':
#             self.data = self.load_data(f"{data_dir}/test")
#             self.ground_truth = self.load_ground_truth(f"{data_dir}/test")
#         elif stage == 'predict':
#             self.data = self.load_data(f"{data_dir}/test")
#         self.view_szes = [dfs[0].shape[1] for dfs in self.data.values()]

#     @staticmethod
#     def normalize_dataset(df):
#         # scaler = StandardScaler() if scaler is None else scaler
#         # standardized_data = scaler.fit_transform(df)
#         scaler = MinMaxScaler()
#         standardized_data = scaler.fit_transform(df)
#         return standardized_data
    
#     def load_data(self, data_dir):
#         view_paths = sorted(glob.glob(data_dir+"/view*"))
#         views_dfs = {view_path[-6:]: None for view_path in view_paths}
#         for view_path in view_paths:
#             p_paths = sorted(glob.glob(view_path+"/*.csv"))
#             view_dfs = []
#             for path in p_paths:
#                 df = pd.read_csv(path, dtype=np.float32)
#                 data = self.normalize_dataset(df)
#                 view_dfs.append(data)
#             views_dfs[view_path[-6:]] = view_dfs
#         # Get length of all mts
#         self.lens = [df.shape[0] // self.window_size for df in views_dfs['view_1']]
#         return views_dfs

#     # def load_data(self, data_dir):
#     # 	view_paths = sorted(glob.glob(data_dir+"/view*"))
#     # 	views_dfs = {view_path[-6:]: None for view_path in view_paths}
#     # 	for view_path in view_paths:
#     # 		p_paths = sorted(glob.glob(view_path+"/*.csv"))
#     # 		view_dfs = []
#     # 		for path in p_paths:
#     # 			df = pd.read_csv(path, dtype=np.float32)
#     # 			data = self.normalize_dataset(df)
#     # 			view_dfs.append(data)
#     # 		views_dfs[view_path[-6:]] = view_dfs
#     # 	# Get length of all mts
#     # 	self.lens = [df.shape[0] // self.window_size for df in views_dfs['view_1']]
#     # 	return views_dfs

#     def load_ground_truth(self, data_dir):
#         paths = sorted(glob.glob(data_dir+"/*.csv"))
#         dfs = [pd.read_csv(path, index_col=False, dtype=np.int32)['is_anomaly'] for path in paths]
#         return dfs

#     def __len__(self):
#         return sum(self.lens)

#     def __getitem__(self, index):
#         i = 0
#         for len_ in self.lens:
#             if index >= len_:
#                 index -= len_
#                 i += 1
#             else:
#                 break
#         X = [torch.tensor(view_dfs[i][index*self.window_size:(index+1)*self.window_size]) for v, view_dfs in self.data.items()]
#         y = torch.tensor(self.ground_truth[i][(index+1)*self.window_size-1]) if self.stage == 'test' else torch.tensor(0)
#         return X, y