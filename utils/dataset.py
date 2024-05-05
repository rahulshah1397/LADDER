#!/usr/bin/env python3
import numpy as np
import toml
import torch
import scipy.integrate as integrate
from itertools import product

config = toml.load("config.toml")


class Samples:
    """ Class implementing sampling from the dataset. We choose K(=pairs) arbit
    rary points from the query set and sample from their joint distribution to
    get one training sample.
    """
    split_set = False
    train_idx, val_idx = None, None

    def __init__(self, mode, pairs = None):
        assert mode in ['train','test','val']
        self.pairs = config['dataset']['PAIRS'] if pairs is None else pairs
        self.mode = mode
        self.X, self.Y, self.flags, self.cov = self._read_data()
        self.Y = self.Y - self.cdm(self.X)
        self.test_idx = np.where(~self.flags)[0]
        self.training_idx = np.where(self.flags)[0]
        if not self.__class__.split_set:
            self.__class__.split_set = True
            self._init_split()
        self.support, self.query = self._get_support_query()
        self.size = config['dataset']['TRAIN_SIZE'] if self.mode == "train" else len(self.query)

    def __len__(self):
        return self.size

    def _init_split(self):
        """ Train validation split method """
        split = np.random.binomial(1,config['dataset']['TRAIN_SPLIT'], len(self.training_idx))
        self.__class__.train_idx = self.training_idx[split.astype(bool)]
        self.__class__.val_idx = self.training_idx[~split.astype(bool)]

    def _get_support_query(self):
        """ Designates the support set (from where training samples are drawn) 
        and query set, for which the model is required to predict values.
        """
        if self.mode == 'test':
            support, query = self.training_idx, self.test_idx
        elif self.mode == 'val':
            support, query = self.train_idx, self.val_idx
        elif self.mode == 'train':
            support, query = self.train_idx, self.train_idx
        return support, query

    def __getitem__(self, key):
        """ Picks a query point from the query set, and pairs-1 points from the 
        support set. Samples from the joint distribution of the support points
        and sorts them according to their z values to create a training sample.
        """
        if key >= len(self):
            raise IndexError(f"Dataset of size {len(self)} has no {key}th position")
        target_idx = self.query[key%len(self.query)]
        target_x, target_y, target_var = self.X[target_idx], self.Y[target_idx], self.cov[target_idx, target_idx]
        idxs = np.random.choice(self.support, self.pairs-1, replace = False)
        _supp = [list(j) for j in self._from_normal_distribution(idxs)]
        _supp.sort(key=lambda x:x[0])
        _supp.append([target_x, np.mean(_supp[-1])])
        return np.array(_supp), target_y, target_var

    def _from_normal_distribution(self, idxs):
        """ Samples from the joint normal distribution of the datapoints designated
        by the indices idxs.
        """
        X = self.X[idxs]
        mean = self.Y[idxs]
        cov = self.submatrix(idxs)
        sample = np.random.multivariate_normal(mean, cov)
        return [(x,y) for x,y in zip(X, sample)]

    def submatrix(self, idxs):
        """ Creates the requisite submatrix from the full covariance matrix
        with the given indices
        Args:
            x : np.ndarray
        Returns:
            res (np.ndarray) : The submatrix with the requisite indices
        """
        res = np.zeros((len(idxs),len(idxs)))
        for (i,num_x), (j,num_y) in product(enumerate(idxs), repeat=2):
            res[i,j] = self.cov[num_x, num_y]
        return res

    @staticmethod
    def cdm(X):
        """ Computes lambda CDM values at given z.

        Args:
            x : [np.ndarray, torch.tensor, float]

        Returns:
            res (np.ndarray, torch.tensor, float) : The lambda CDM predicted 
                                                    value of m
        """
        def _cdm(x):
            Hz = lambda z:config['cdm']['H0']*np.sqrt(
                    config['cdm']['om0']*(1+z)**3 + 1.-config['cdm']['om0'])
            integrand = lambda z: config['cdm']['c']/Hz(z)
            dc2,dc1 = integrate.quad(integrand, 0., x)
            dL = (1+x)*(dc2-dc1)
            return 5.0*np.log10(dL) + 25. + config['cdm']['MB']
        if isinstance(X, np.ndarray):
            res = np.array([_cdm(x) for x in X])
        elif isinstance(X, torch.Tensor):
            res = torch.tensor([_cdm(x) for x in X])
        else:
            res = _cdm(X)
        return res

    @staticmethod
    def _read_data():
        """ Reads data fram CSV file and converts to numpy arrays. Also
        initializes train test splits from the data file.

        Args:
            None

        Returns:
            X (np.array): the redshift values.
            Y (np.array): the corresponding m values.
            mark (np.array): boolean array indicating train/test.
            cov (np.array): covariance matrix.
        """
        def _process(row):
            row = row.split(" ")
            return float(row[1]), float(row[2]), float(row[3]), (row[4]=="True")
        cov_matrix = np.loadtxt(config['files']['COV'], skiprows=1)
        cov_matrix = cov_matrix.reshape(int(np.sqrt(cov_matrix.shape[0])), int(np.sqrt(cov_matrix.shape[0])))
        cov_matrix = (cov_matrix + cov_matrix.T)/2
        with open(config['files']['DATA'], 'r') as f:
            raw = f.readlines()[1:]
        rows = [i.strip() for i in raw]
        X, Y, dY, mark = [],[],[],[]
        for row in rows:
            x, y, dy, flag = _process(row)
            X.append(x)
            Y.append(y)
            dY.append(dy)
            mark.append(flag)
        X, Y, dY, mark = np.array(X),np.array(Y),np.array(dY), np.array(mark, dtype=bool)
        for i in range(len(dY)):
            cov_matrix[i][i] += dY[i]**2
        return X, Y, mark, cov_matrix


class Dataset(torch.utils.data.Dataset):
    """ Creates torch tensors from the Samples given by samples. """
    def __init__(self, mode, pairs = None):
        self.samples = Samples(mode = mode, pairs = pairs)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, key):
        feature, target_mean, target_var = self.samples[key]
        return (
                    torch.tensor(feature, dtype = torch.float32),
                    torch.tensor(target_mean, dtype = torch.float32),
                    torch.tensor(target_var, dtype = torch.float32)
                )
