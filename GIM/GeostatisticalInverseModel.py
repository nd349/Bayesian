# -*- coding: utf-8 -*-
# @Author: nikhildadheech
# @Date:   2022-08-28 19:19:58
# @Last Modified by:   nikhildadheech
# @Last Modified time: 2023-03-25 19:06:55

import numpy as np
import pandas as pd
from scipy.sparse import csc_matrix, csr_matrix, coo_matrix
from scipy.sparse.linalg import inv
import time
import datetime
# import torch
import netCDF4 as nc
from tqdm import tqdm


from config import *
from GIM.Covariance import *
from Utils.HQ_HQHT import HQ, HQHT, computeQHTeta
from Utils.reshape_matrices import *
from GIM.GIM_config import *


class GIM():

    """
        Inversion module with full prior covariance metrices
        object: <class>
        
    """
    def __init__(self, H, X, Y, So, observation_dict):
        """
        Initializing inversion module

        Arguments:
            H: <1-D array>
            X: <1-D array>
            Y: <1-D array>
            So: <2-D array>
        returns:
            None
        """
        self.mode = mode
        self.emulator = emulator
        self.full_prior = full_prior
        self.sparse = sparse
        self.ems_uncert = ems_uncert
        self.minUncert = minUncert
        self.device = device
        self.tau_day = tau_day
        self.tau_hr = tau_hr
        self.tau_len = tau_len
        self.fsigma = fsigma
        self.Sa_xy_file = Sa_xy_file
        self.observation_dict = observation_dict
        
        if not self.full_prior:
            raise Exception(f"diag_prior is expected to be True but found this instead: {diag_prior}")

        if self.sparse:
            print("Forming sparse matrices ....")
            print("H")
            self.H = csc_matrix(H)
            self.H_array = H
            print("X")
            self.X_pri = csc_matrix(X)
            self.X_pri_array = X
            print("Y")
            self.Y = csc_matrix(Y)
            self.Y_array = Y
            self.Sa_t = build_temporal(self.tau_day, self.tau_hr, self.X_pri_array, m) # numpy array
        else:
            self.H = H
            self.X_pri = X
            self.Y = Y
            self.Sa_t = build_temporal(self.tau_day, self.tau_hr, self.X_pri, m)


        self.p = p_GIM
        self.landmask = landmask
        if self.landmask:
            if not landmask_load:
                self.form_landmask_vector()
            elif landmask_load:
                # you can also load a landmask_filter here by giving a path
                pass

        # Forming X_GIM with p
        self.form_X_GIM()  # Creates self.X_GIM variable which will be used in the inversion


        # self.Sa_xy = self.form_spatial_covariance()
        self.Sa_xy = self.form_spatial_covariance(X) # csr array or csc matrix?
        
        self.So = So # csc matrix
        self.X_hat = None

        # print(f"Type of H:{type(self.H)}")
        # print(f"Type of X :{type(self.X_pri)}")
        # print(f"Type of Y:{type(self.Y)}")
        # print(f"Type of Sa_t:{type(self.Sa_t)}")
        # print(f"Type of Sa_xy:{type(self.Sa_xy)}")
        # print(f"Type of So:{type(self.So)}")


    def form_spatial_covariance(self, X):
        """
        Form spatial covariance matrix

        Arguments:
            X: <1-D array>
        returns:
            Sa_xy_covariance: <csc matrix>
        """
        # import pdb; pdb.set_trace()
        print ("Forming spatial covariance")
        Sa_xy = load_Sa_xy(self.Sa_xy_file)

        nEms = int(self.X_pri.shape[0]/m)
        nG = m
        emsAll = np.zeros((nEms, nG), dtype=np.float32)
        variance_xy = np.zeros((nG, nG), dtype=np.float32)
        for i in tqdm(range(nG)):
            emsAll[:, i] = np.array([X[j*m:(j+1)*m, 0][i] for j in range(nEms)])
            variance_xy[i, i] = np.var(emsAll[:, i])

        print("Converting variance into sparse")
        variance_xy = csc_matrix(variance_xy)
        variance_xy_sqrt = csc_matrix.sqrt(variance_xy)
        print("Computing covariance matrix")
        Sa_xy_covariance = np.sqrt(self.fsigma)*(variance_xy_sqrt.multiply(Sa_xy.multiply(variance_xy_sqrt)))
        # Sa_xy = np.sqrt(self.fsigma)*(csr_matrix.dot(csr_matrix.sqrt(variance_xy), csr_matrix.dot(Sa_xy), csr_matrix.sqrt(variance_xy)))
        return Sa_xy_covariance

    def get_concentrations(self):
        y_prior = np.dot(self.H_array, self.X_pri_array)
        y_posterior = np.dot(self.H_array, self.original_X_hat)
        return np.array(y_posterior), np.array(y_prior)


    def remove_padding(self, X_hat):
        """
        Removes padding from the solution (back hours)

        Arguments:
            X_hat: <1-D array>
        returns:
            X_hat: <1-D array>
        """
        print("Type of X_hat:", type(X_hat))
        X_hat = X_hat[back_hours*m:(X_hat.shape[0]-back_hours*m)]
        return X_hat


    def form_X_GIM(self):
        if self.p >=2:
            try:
                self.X_GIM = np.ones((self.H.shape[1], self.p))
                self.X_GIM[:, 1:] = self.X_pri_array
                if self.landmask:
                    self.X_GIM[:, 0] = self.landmask_vector[:, 0]
            except Exception as e:
                print(e)
                raise Exception("Not able to generate X_GIM which may need X (prior), landmask or any other values")
        elif self.p ==1:
            self.X_GIM = np.ones((self.H.shape[1], self.p))
            if self.landmask:
                self.X_GIM[:, 0] = self.landmask_vector


    def form_landmask_vector(self):
        temp = np.zeros((nrow, ncol))
        for idx in range(int(self.H.shape[1]/m)):
            temp += self.X_pri_array[idx*m:(idx+1)*m].reshape(nrow, ncol, order='F')

        temp[np.where(temp!=0)] = 1
        self.landmask_array = temp
        self.landmask_vector = np.zeros((self.H.shape[1], 1))
        for idx in range(int(self.H.shape[1]/m)):
            self.landmask_vector[idx*m:(idx+1)*m] = temp.reshape(m, 1, order='F')
        return

    def formAB(self, HQHT_R, HX):
        n = self.Y.shape[0]
        p = self.p

        A = np.zeros((n+p, n+p))
        B = np.zeros((n+p, 1))

        A[:n, :n] = HQHT_R
        A[n:, :n] = HX.T
        A[:n, n:] = HX

        B[:n, 0] = self.Y_array[:, 0]

        return A, B, n



    def invert(self):
        """
        Invert for posterior solution using Bayesian inference method

        Arguments:
            None
        returns:
            self.X_hat: <2-D array>
        """
        print("Inversion is starting .....")
        print(f"Size of H: {self.H.shape}")
        print(f"Size of X prior: {self.X_pri.shape}")
        print(f"Size of Y: {self.Y.shape}")
        print(f"Size of So: {self.So.shape}")
        print(f"Size of Sa_t: {self.Sa_t.shape}")
        print(f"Size of Sa_xy: {self.Sa_xy.shape}")
        print(f"Size of X_GIM: {self.X_GIM.shape}")

        start = time.time()
        if self.device == 'cpu':
            HQ_GIM = HQ(self.H_array, self.Sa_t, self.Sa_xy, parallel=hq_parallel)
            HQHT_GIM = HQHT(HQ_GIM, self.H_array, self.Sa_t, self.Sa_xy)
            HQHT_R = HQHT_GIM + self.So
            HX = np.dot(self.H_array, self.X_GIM)

            A, B, n = self.formAB(HQHT_R, HX)  # n is number of observations

            weights = np.linalg.solve(A, B)

            beta = weights[n:]
            eta = weights[:n]

            self.beta = beta
            self.eta = eta

            QHTeta = computeQHTeta(self.H_array, self.Sa_t, self.Sa_xy, eta)

            self.QHTeta = QHTeta

            s_hat = np.dot(self.X_GIM, beta) + QHTeta

            self.original_X_hat = s_hat
            self.X_hat = self.remove_padding(s_hat)
            print(f"Time taken for inversion: {time.time()-start} seconds")
            return self.X_hat

            
        elif self.device == 'cuda':
            pass



    def save_solution(self):
        """
        Saving posterior solution

        Arguments:
            None
        returns:
            None
        """
        X_hat = self.X_hat
        y_posterior, y_prior = self.get_concentrations()
        X_hat_grid = np.zeros((int(X_hat.shape[0]/m), nrow, ncol))
        solution_date_range = pd.date_range(start=start_time, end=end_time-datetime.timedelta(hours=1), freq='1h')
        
        for idx in range(X_hat_grid.shape[0]):
            # X_hat_grid[idx, :, :] = make_grid_2d_column(X_hat[idx*m:(idx+1)*m, 0], nrow, ncol)
            X_hat_grid[idx, :, :] = X_hat[idx*m:(idx+1)*m, 0].reshape(nrow, ncol, order='F')

        print("Saving output at:", output_directory)
        
        year = str(start_time.year)
        month = str(start_time.month)
        day = str(start_time.day)
        

        if len(month) == 1:
            month = "0"+month
        if len(day) == 1:
            day = "0"+day
        

        if emulator:
            conc_file = f"{output_directory}emulator_observations_prior_posterior_{year}x{month}x{day}.nc"
        else:
            conc_file = f"{output_directory}STILT_observations_prior_posterior_{year}x{month}x{day}.nc"

        conc_nc = nc.Dataset(conc_file, "w", format="NETCDF4")
        conc_nc.createDimension("nobs", self.Y.shape[0])
        obs_prior = conc_nc.createVariable("obs_prior", "f8", ("nobs"))
        obs_posterior = conc_nc.createVariable("obs_posterior", "f8", ("nobs"))
        obs_actual = conc_nc.createVariable("obs_actual", "f8", ("nobs"))

        obs_prior[:] = y_prior
        obs_posterior[:] = y_posterior
        obs_actual[:] = self.Y_array

        obs_year = conc_nc.createVariable("obs_year", "f8", ("nobs"))
        obs_month = conc_nc.createVariable("obs_month", "f8", ("nobs"))
        obs_day = conc_nc.createVariable("obs_day", "f8", ("nobs"))
        obs_hour = conc_nc.createVariable("obs_hour", "f8", ("nobs"))

        obs_lats = conc_nc.createVariable("obs_lat", "f8", ("nobs"))
        obs_lons = conc_nc.createVariable("obs_lon", "f8", ("nobs"))

        obs_year[:] = [datetime.datetime.strftime(term['time'], '%Y%m%d%H')[:4] for term in list(self.observation_dict.values())]
        obs_month[:] = [datetime.datetime.strftime(term['time'], '%Y%m%d%H')[4:6] for term in list(self.observation_dict.values())]
        obs_day[:] = [datetime.datetime.strftime(term['time'], '%Y%m%d%H')[6:8] for term in list(self.observation_dict.values())]
        obs_hour[:] = [datetime.datetime.strftime(term['time'], '%Y%m%d%H')[8:] for term in list(self.observation_dict.values())]

        obs_lats[:] = [term['lat'] for term in list(self.observation_dict.values())]
        obs_lons[:] = [term['lon'] for term in list(self.observation_dict.values())]

        conc_nc.close()
        print(f"{conc_file} has been saved ....")

        for idx, timestamp in tqdm(enumerate(solution_date_range)):
            year = str(timestamp.year)
            month = str(timestamp.month)
            day = str(timestamp.day)
            hour = str(timestamp.hour)
            if len(month)==1:
                month = '0'+month
            if len(day) == 1:
                day = '0'+day
            if len(hour) == 1:
                hour = '0'+hour
            timestamp = f"{year}{month}{day}{hour}"
            if emulator:    
                file = f"{output_directory}emulator_posterior_{location}_{year}x{month}x{day}x{hour}.ncdf"
            else:
                file = f"{output_directory}STILT_posterior_{location}_{year}x{month}x{day}x{hour}.ncdf"
            flux = X_hat_grid[idx, :, :]
            # print(file)


            out_nc = nc.Dataset(file, "w", format='NETCDF4')
            out_nc.createDimension("lat", nrow)
            out_nc.createDimension("lon", ncol)
            out_nc.createDimension("info", 1)
            
            lat = out_nc.createVariable("lat", "f8", ("lat",))
            lon = out_nc.createVariable("lon", "f8", ("lon",))
            
            lat[:] = lats
            lon[:] = lons
            
            soln = out_nc.createVariable("flux", "f8", ("lat", "lon"))
            soln[:,:] = flux
            out_nc.close()
            print(f"{file} has been saved ...")