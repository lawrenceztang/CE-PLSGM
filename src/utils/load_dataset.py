import pickle

import numpy as np

import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset
from torch.utils.data.dataset import Subset
from torch.utils.data import ConcatDataset

from sklearn.datasets import load_boston, fetch_california_housing, fetch_covtype, fetch_kddcup99
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split



def softplus_np(x):
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0)


class My_Dataset(Dataset):
    def __init__(self, train):
        self._y_datatype = self._get_y_datatype()
        X, y = self._load_dataset()
        
        symb_attrs_indices = self._get_symbolic_attrs_indices()
        for ind in symb_attrs_indices:
            le = LabelEncoder()
            le.fit(X[:, ind])
            X[:, ind] = le.transform(X[:, ind])
        if self._y_datatype == torch.int64:
            le = LabelEncoder()
            le.fit(y)
            y = le.transform(y)
        X = X.astype(np.float32)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=0) # random seed is fixed
        scaler = StandardScaler()
        cont_attrs_indices = [i for i in range(X.shape[1]) if i not in symb_attrs_indices]
        scaler.fit(X_train[:, cont_attrs_indices])
        X_train[:, cont_attrs_indices] = scaler.transform(X_train[:, cont_attrs_indices])
        X_test[:, cont_attrs_indices] = scaler.transform(X_test[:, cont_attrs_indices])
        if train:
            self.X = torch.from_numpy(X_train).to(torch.float)
            self.y = torch.from_numpy(y_train).to(self._y_datatype)
        else:
            self.X = torch.from_numpy(X_test).to(torch.float)
            self.y = torch.from_numpy(y_test).to(self._y_datatype)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    
    def _load_dataset(self):
        raise NotImplementedError

    def _get_y_datatype(self):
        raise NotImplementedError

    def _get_symbolic_attrs_indices(self):
        raise NotImplementedError

            
class California_Housing(My_Dataset):
    def _load_dataset(self):
        X, y = fetch_california_housing(return_X_y=True)
        return X, y

    def _get_y_datatype(self):
        return torch.float

    def _get_symbolic_attrs_indices(self):
        return []


class California_Housing(My_Dataset):
    def _load_dataset(self):
        X, y = fetch_california_housing(return_X_y=True)
        y = y / np.absolute(y).max()
        return X, y

    def _get_y_datatype(self):
        return torch.float

    def _get_symbolic_attrs_indices(self):
        return []


class Gas_Turbine(My_Dataset):
    def _load_dataset(self):
        path = "data/Gas_Turbine/data.pickle"
        with open(path, "rb") as f:
            X, y = pickle.load(f)
        y = y / np.absolute(y).max()
        return X, y

    def _get_y_datatype(self):
        return torch.float

    def _get_symbolic_attrs_indices(self):
        return []


class BlogFeedback(My_Dataset):
    def _load_dataset(self):
        path = "data/BlogFeedback/data.pickle"
        with open(path, "rb") as f:
            X, y = pickle.load(f)
        y = y / np.absolute(y).max()
        assert X.shape[1] == 280
        return X, y

    def _get_y_datatype(self):
        return torch.float

    def _get_symbolic_attrs_indices(self):
        return list(range(262, 269))
    

    
def load_dataset(dataset_name):
    if "california_housing" in dataset_name:
        train_data = California_Housing(train=True)
        test_data = California_Housing(train=False)
        n_classes = 1
        n_dims = 8
    elif "gas" in dataset_name:
        train_data = Gas_Turbine(train=True)
        test_data = Gas_Turbine(train=False)
        n_classes = 1
        n_dims = 9
    elif "blog" in dataset_name:
        train_data = BlogFeedback(train=True)
        test_data = BlogFeedback(train=False)
        n_classes = 1
        n_dims = 280
    else:
        assert False
    if dataset_name in []:
        task_type = "classification"
    elif dataset_name in ["california_housing", "gas", "blog"]:
        task_type = "regression"
    else:
        assert False

    return train_data, test_data, n_classes, n_dims, task_type
