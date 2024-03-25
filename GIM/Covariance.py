# -*- coding: utf-8 -*-
# @Author: nikhildadheech
# @Date:   2022-08-25 12:14:36
# @Last Modified by:   nikhildadheech
# @Last Modified time: 2022-10-11 15:18:48


import pickle
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from config import *
from scipy.sparse import csc_matrix, csr_matrix, coo_matrix


Sa_xy_file = "data/Sa_xy_corrcoef_emulator.pkl"


def load_Sa_xy(file):
	"""
        Load spatial correlation matrix

        Arguments:
            file: <str>
        returns:
            Sa_xy: <csc matrix>
    """
	with open(file, 'rb') as open_file:
		Sa_xy = pickle.load(open_file)
	Sa_xy = csc_matrix(Sa_xy)
	return Sa_xy

def build_temporal(tau_day, tau_hr, x_pri, m):
    """
        Build the temporal covariance matrix

        Arguments:
            tau_day: <float>
            tau_hr: <float>
            x_pri: <1D array>
            m: <int>
        returns:
    """
    print("Forming temporal covariance")
    nEms = int(x_pri.shape[0]/m)
    nG = m
    tempA = np.zeros((nG))
    tempB = np.zeros((nG))
    Sa_t = np.zeros((nEms, nEms))
    variance_temporal = np.zeros((nEms, nEms), dtype=np.float32)
    lowBoundCount = 0
    for i in tqdm(range(nEms)):
        tempA = x_pri[i*m:(i+1)*m]
        variance_temporal[i,i] = np.var(tempA)
        for j in range(i, nEms):
            tempB = x_pri[j*m:(j+1)*m]
            hours_apart = j-i
            days_apart = hours_apart/24
            hours_apart = 12   - abs(12   - np.mod(hours_apart,24))
            temp_hours = np.exp(-abs(hours_apart)/tau_hr)
            temp_days = np.exp(-abs(days_apart)/tau_day)
            # print(temp_hours, temp_days)
            cor = np.corrcoef(tempA[:, 0], tempB[:, 0])[0,1]
            # print(i, j, hours_apart, days_apart, temp_hours, temp_days, tempA.shape, tempB.shape, cor)
            sig_val = cor*temp_hours*temp_days
            if sig_val > lowBound:
                Sa_t[i, j] = sig_val
                Sa_t[j, i] = sig_val
            else:
                lowBoundCount += 1
                # print(i, j)
            #     Sa_t[i, j] = 10**-7
            #     Sa_t[j, i] = 10**-7
    print("lowBoundCount:", lowBoundCount)
    Sa_t = compute_covariance(Sa_t, variance_temporal, fsigma=fsigma)
    return Sa_t # need to convert this to covariance


def compute_covariance(Sa_t, variance_t, fsigma=fsigma):
    """
        Compute the temporal covariance matrix

        Arguments:
            Sa_t: <2D array>
            variance_t: <2D array>
            fsigma: <float>
        returns:
            Sa_t: <2D array>
    """
    Sa_t = np.sqrt(fsigma)*np.dot(np.sqrt(variance_t), np.dot(Sa_t, np.sqrt(variance_t)))
    return Sa_t

def plot_temporal_matrix(Sa_t, output_plot_file, title=''):
    """
        Plot the temporal covariance matrix

        Arguments:
            Sa_t: <2D array>
            output_plot_file: <str>
            title: <str>
        returns:
    """
    Sa_t[np.where(Sa_t==0)] = 10**-20
    index = [i for i in range(Sa_t.shape[0])]
    h = plt.pcolor(index, index, Sa_t)
    plt.colorbar(h)
    plt.xlabel("Hours")
    plt.ylabel("Hours")
    if title:
        plt.title(title)
    plt.savefig(output_plot_file)
    plt.close()

def plot_variation_with_one_time(Sa_t, index, output_plot_file, title='', label=''):
    """
        Plot the temporal covariance matrix with one time point

        Arguments:
            Sa_t: <2D array>
            index: <int>
            output_plot_file: <str>
            title: <str>
            label: <str>
        returns:
    """
    value = Sa_t[index, :]
    if label:
        plt.plot(value, label=label)
        plt.legend()
    else:
        plt.plot(value)
    plt.xlabel("Hours")
    plt.ylabel("Covariance")
    if title:
        plt.title(title)
    plt.savefig(output_plot_file)


if __name__ == '__main__':
    X = None
    Sa_t, variance_t = build_temporal(tau_day, tau_hr, X, m)
    plot_temporal_matrix(Sa_t, "plots/temporal/temporal_output.png", title='Temporal covariance matrix')
    index = 0
    plot_variation_with_one_time(Sa_t, index, f"plots/temporal/temporal_{index}_variation.png", title=f'Temporal covariance with respect of 00:00 hour')