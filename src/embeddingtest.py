import torch
import numpy as np
import pandas as pd

class MMDTest:
    def __init__(self, features_X, features_Y, n_perm=1000):

        self.n_perm = n_perm
        self.features_X = features_X
        self.features_Y = features_Y

    def _compute_mmd(self, features_X, features_Y):

        mean_fX = features_X.mean(0)
        mean_fY = features_Y.mean(0)
        D = mean_fX - mean_fY
        statistic = np.linalg.norm(D)**2
        return statistic

    def _compute_p_value(self):

        # compute real test statistic
        stat = self._compute_mmd(self.features_X, self.features_Y)
        n, m = len(self.features_X), len(self.features_Y)
        l = n + m
        features_Z = np.vstack((self.features_X, self.features_Y))

        # compute null samples
        resampled_vals = np.empty(self.n_perm)
        for i in range(self.n_perm):
            index = np.random.permutation(l)  # it permutes indices from 0 to l
            feats_X, feats_Y = features_Z[index[:n]], features_Z[index[n:]]
            resampled_vals[i] = self._compute_mmd(feats_X, feats_Y)

        resampled_vals.sort()
        # p_val = np.mean(stat < resampled_vals)
        p_val = (np.sum(stat <= resampled_vals)+1)/(self.n_perm+1)
        return p_val