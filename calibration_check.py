import numpy as np
import datetime # ; import time
import numpy as np
import pandas as pd
import sys, os
from Utils.filter_query import *
from Utils.reshape_matrices import *
from Utils.readData import *
from sklearn.model_selection import train_test_split
from joblib import Parallel, delayed
from tqdm import tqdm
from scipy.sparse import csc_matrix, csr_matrix, coo_matrix
# from Utils.getData import *
# from config import *
# from fullCovariance.InversionFullCovariance import InversionFullPrior
from fullCovariance.SpatialCovariance import *
from fullCovariance.TemporalCovariance import *
from Utils.HQ_HQHT import HQ, HQHT

def get_background_error(data, bkg_conc_noaa, bkg_conc_nasa, bkg_conc_ameriflux, bkg_err_noaa, bkg_err_nasa, bkg_err_ameriflux):
    """
        Get the background error

        Arguments:
            data: <netcdf object>
            bkg_conc_noaa: <float>
            bkg_conc_nasa: <float>
            bkg_conc_ameriflux: <float>
            bkg_err_noaa: <float>
            bkg_err_nasa: <float>
            bkg_err_ameriflux: <float>
        returns:
            <float>, <float>, <str>
    """
    source = ''
    bkg_conc = bkg_conc_noaa
    bkg_err = max(bkg_err_nasa, bkg_err_noaa)
    if bkg_err_nasa > bkg_err_noaa:
        source = 'nasa'
    else:
        source = 'noaa'
    if bkg_err == -999.0: #nan value auto fill in netcdf file
        if bkg_err_nasa == -999.0 and bkg_err_noaa == -999.0:
            bkg_err = np.nan
        elif bkg_err_nasa == -999.0:
            source = 'noaa'
            bkg_err = bkg_err_noaa
        elif big_err_noaa == -999.0:
            source = 'nasa'
            bkg_err = bkg_err_nasa
    
    ameriflux_lon = -121.8
    ameriflux_lat =   38.2
    maxDist       =   25.0 # maximum allowable distance from the AmeriFlux site
    end_lon = np.float32(np.array(data['end_lon'])[0])
    end_lat = np.float32(np.array(data['end_lat'])[0])
    end_agl = np.float32(np.array(data['end_agl'])[0])
    amf_lon = np.float32(np.array(data['ameriflux_lon'])[0])
    amf_lat = np.float32(np.array(data['ameriflux_lat'])[0])
    amf_agl = np.float32(np.array(data['ameriflux_agl'])[0])
    amf_time = np.float32(np.array(data['ameriflux_julian'])[0])
    amf_dist = 110*np.sqrt((amf_lon-ameriflux_lon)**2 + (amf_lat-ameriflux_lat)**2)
    # amfUSE  = amfDist .< maxDist
    if amf_dist < maxDist:
        source = 'ameriflux'
        bkg_conc = bkg_conc_ameriflux
        bkg_err = max(bkg_err, bkg_err_ameriflux)
    return bkg_conc, bkg_err, source

def parse_obs_info(file):
    """
        Parse the observation file details (timestamp, location etc.)

        Arguments:
            file: <str>
            
        returns:
            <list>
    """
    [__, timestamp, receptor_lon, receptor_lat, receptor_agl] = file.replace('.nc', '').split("_")
    return [__, timestamp, receptor_lon, receptor_lat, receptor_agl]

def parse_ems_info(file):
    """
        Parse the emission file details (timestamp etc.)

        Arguments:
            file: <str>

        returns:
            <list>
    """
    trimmed_file = file.split("/")[-1].replace("_", "x")
    [_, year, month, day, hour] = trimmed_file.replace('.ncdf', '').split("x")
    return [_, year, month, day, hour]

def get_len(coords_1, coords_2):
    """
        Compute the length between two coordinates

        Arguments:
            coords_1: <list>
            coords_2: <list>

        returns:
            <float>
    """
    lat1 = coords_1[0]*np.pi/180
    lon1 = coords_1[1]*np.pi/180
    lat2 = coords_2[0]*np.pi/180
    lon2 = coords_2[1]*np.pi/180
    R = 6371e3
    # a = sin²(Δφ/2) + cos φ1 ⋅ cos φ2 ⋅ sin²(Δλ/2)
    # c = 2 ⋅ atan2( √a, √(1−a) )
    # d = R ⋅ c
    a = np.sin((lat1-lat2)/2)**2 + np.cos(lat1)*np.cos(lat2)*(np.sin((lon1-lon2)/2)**2)
    c = 2*np.arctan2(np.sqrt(a), np.sqrt(1-a))
    return R*c/1000 #km


def create_weights(x=1, r=0.5):
    n = back_hours+1
    a = x*(1-r)/(1-r**n)
    weight_list = []
    weight_list.append(a)
    term = a
    for idx in range(n-1):
        term = term*r
        weight_list.append(term)
    weight_list.reverse()
    return weight_list


class read_data():

    def __init__(self, m_start, m_end, cross_validation):
        emission_filtered_files, emission_files_df = filter_emissions(emission_files, [m_start, m_end])
        foot_filtered_files, foot_files_df = filter_obs(footprint_files, Inv_lonLim, Inv_latLim, agl_domain='')
        
        foot_files_time_df = foot_files_df[(foot_files_df['time']>=start_time)&(foot_files_df['time']<end_time)]
        
        

        self.cross_validation = cross_validation
        if cross_validation:
            domain_foot_files_train, domain_foot_files_validation = train_test_split(foot_files_time_df, test_size=cross_validation_fraction, random_state=42)
            domain_foot_files = list(domain_foot_files_train['file'])
            domain_foot_files_valid = list(domain_foot_files_validation['file'])
            self.validation_dict = {}
            self.H_valid = np.zeros((len(domain_foot_files_valid), date_range.shape[0]*m), dtype=np.float32)
            self.H_check_valid = np.zeros((len(domain_foot_files_valid), date_range.shape[0]*m), dtype=np.float32)
            self.Y_valid = np.zeros((len(domain_foot_files_valid), 1), dtype=np.float32)
            self.BKG_valid = np.zeros((len(domain_foot_files_valid), 1), dtype=np.float32)
            print("Training size:", len(domain_foot_files))
            print("Validation size:", len(domain_foot_files_valid))
        else:
            print("Observations have not been kept out for the validation ....(everything is used in the training)")
            domain_foot_files = list(foot_files_time_df['file'])


        self.observation_dict = {}
        self.H = np.zeros((len(domain_foot_files), date_range.shape[0]*m), dtype=np.float32)
        self.H_check = np.zeros((len(domain_foot_files), date_range.shape[0]*m), dtype=np.float32)
        self.Y = np.zeros((len(domain_foot_files), 1), dtype=np.float32)
        self.BKG = np.zeros((len(domain_foot_files), 1), dtype=np.float32)
        self.So_d = np.zeros((len(domain_foot_files), 1), dtype=np.float32)
        self.X = np.zeros((date_range.shape[0]*m, 1), dtype=np.float32)
        self.nObs = self.So_d.shape[0]
        self.So = np.zeros((self.nObs, self.nObs), dtype=np.float32)
        self.receptor_list = []

        # print(domain_foot_files_validation)
        self.fill_obs_parallel(domain_foot_files, domain_foot_files_valid)
        self.load_prior_emissions(emission_filtered_files)
        self.fill_observation_covariance_matrix()
        


    def fill_obs_data(self, idx, foot_file, mode='integrated'):

        H = self.H
        Y = self.Y
        So_d = self.So_d
        observation_dict = self.observation_dict
        H_check = self.H_check
        BKG = self.BKG
        
        [__, timestamp, receptor_lon, receptor_lat, receptor_agl] = parse_obs_info(foot_file)
        receptor = [clon, clat, timestamp, receptor_lon, receptor_lat, idx]
        year = int(timestamp[0:4])
        month = int(timestamp[4:6])
        day = int(timestamp[6:8])
        hour = int(timestamp[8:])
        time_foot = datetime.datetime(year, month, day, hour)
        index = time_dict[time_foot]
        observation_dict[idx] = {
            "time":time_foot,
            "lon":float(receptor_lon),
            "lat":float(receptor_lat),
            "agl":float(receptor_agl)
        }
    
    
        foot_data = nc.Dataset(foot_file)
        bkg_conc_noaa = np.float32(np.array(foot_data['bkg_co2_NOAA'])[0])
        bkg_conc_nasa = np.float32(np.array(foot_data['bkg_co2_NASA'])[0])
        bkg_conc_ameriflux = np.float32(np.array(foot_data['ameriflux_co2'])[0])
        bkg_err_noaa = np.float32(np.array(foot_data['bkg_err_NOAA'])[0])
        bkg_err_nasa = np.float32(np.array(foot_data['bkg_err_NASA'])[0])
        bkg_err_ameriflux = np.float32(np.array(foot_data['ameriflux_err'])[0])
        bkg_conc, bkg_error, source = get_background_error(foot_data, bkg_conc_noaa, bkg_conc_nasa, bkg_conc_ameriflux, bkg_err_noaa, bkg_err_nasa, bkg_err_ameriflux)
        Y[idx, 0] = np.float32(np.array(foot_data['co2'])[0]) - bkg_conc
        BKG[idx, 0] = bkg_conc
        # if Y[idx, 0] < 0:
        #     print(foot_file, np.float32(np.array(foot_data['co2'])[0]), bkg_conc, source)
        obs_error = np.float32(np.array(foot_data['co2_err'])[0])
        mod_error = model_error[time_foot.hour]
        So_d[idx, 0] = obs_error**2+bkg_error**2+mod_error**2
        if not emulator:
            if mode == 'integrated':
                foot = np.float32(np.nansum(np.array(foot_data['foot']), axis=0))[clat_index-200:clat_index+200, clon_index-200:clon_index+200]
                H[idx, index*m:(index+1)*m] = flatten_2d_column(foot)
            elif mode == 'integrated_average':
                foot = np.float32(np.average(np.array(foot_data['foot']), axis=0))[clat_index-200:clat_index+200, clon_index-200:clon_index+200]
                # To verify the code (code testing): foot_check and H_check
                foot_check = np.float32(np.nansum(np.array(foot_data['foot']), axis=0))[clat_index-200:clat_index+200, clon_index-200:clon_index+200]
                H_check[idx, index*m:(index+1)*m] = flatten_2d_column(foot_check)
    
                # resolved_time_list = date_range[date_range<=time_foot][-foot.shape[0]:]
                resolved_time_list = date_range[date_range<=time_foot][-(back_hours+1):]
                for jdx, time_hour in enumerate(resolved_time_list):
                    m_index = time_dict[time_hour]
                    H[idx, m_index*m:(m_index+1)*m] = flatten_2d_column(foot)
            elif mode == 'integrated_decayed':
                # declare the weights
                weight_list = create_weights()
                foot = np.float32(np.nansum(np.array(foot_data['foot']), axis=0))[clat_index-200:clat_index+200, clon_index-200:clon_index+200] #compressed footprints
                H_check[idx, index*m:(index+1)*m] = flatten_2d_column(foot)
                resolved_time_list = date_range[date_range<=time_foot][-(back_hours+1):]
                for jdx, time_hour in enumerate(resolved_time_list):
                    m_index = time_dict[time_hour]
                    H[idx, m_index*m:(m_index+1)*m] = flatten_2d_column(foot*weight_list[jdx])
                # import pdb; pdb.set_trace()
            elif mode == 'resolved':
                foot = np.float32(np.array(foot_data['foot']))[:, clat_index-200:clat_index+200, clon_index-200:clon_index+200]
                resolved_time_list = date_range[date_range<=time_foot][-foot.shape[0]:]
                for jdx, time_hour in enumerate(resolved_time_list):
                    m_index = time_dict[time_hour]
                    H[idx, m_index*m:(m_index+1)*m] = flatten_2d_column(foot[jdx, :, :])
        foot_data.close()

        # self.H = H.copy()
        # self.Y = Y.copy()
        # self.So_d = So_d.copy()
        # self.observation_dict = observation_dict.copy()
        # self.H_check = H_check.copy()
        # self.BKG = BKG.copy()
        return receptor


    def fill_obs_validation(self, idx, foot_file, mode='integrated'):

        H_valid = self.H_valid
        Y_valid = self.Y_valid
        validation_dict = self.validation_dict
        BKG_valid = self.BKG_valid
        H_check_valid = self.H_check_valid
        # print(foot_file)
        [__, timestamp, receptor_lon, receptor_lat, receptor_agl] = parse_obs_info(foot_file)
        receptor = [clon, clat, timestamp, receptor_lon, receptor_lat, idx]
        year = int(timestamp[0:4])
        month = int(timestamp[4:6])
        day = int(timestamp[6:8])
        hour = int(timestamp[8:])
        time_foot = datetime.datetime(year, month, day, hour)
        index = time_dict[time_foot]
        validation_dict[idx] = {
            "time":time_foot,
            "lon":float(receptor_lon),
            "lat":float(receptor_lat),
            "agl":float(receptor_agl)
        }
    
        foot_data = nc.Dataset(foot_file)
        bkg_conc_noaa = np.float32(np.array(foot_data['bkg_co2_NOAA'])[0])
        bkg_conc_nasa = np.float32(np.array(foot_data['bkg_co2_NASA'])[0])
        bkg_conc_ameriflux = np.float32(np.array(foot_data['ameriflux_co2'])[0])
        bkg_err_noaa = np.float32(np.array(foot_data['bkg_err_NOAA'])[0])
        bkg_err_nasa = np.float32(np.array(foot_data['bkg_err_NASA'])[0])
        bkg_err_ameriflux = np.float32(np.array(foot_data['ameriflux_err'])[0])
        bkg_conc, bkg_error, source = get_background_error(foot_data, bkg_conc_noaa, bkg_conc_nasa, bkg_conc_ameriflux, bkg_err_noaa, bkg_err_nasa, bkg_err_ameriflux)
        Y_valid[idx, 0] = np.float32(np.array(foot_data['co2'])[0]) - bkg_conc
        BKG_valid[idx, 0] = bkg_conc
    
        # H_valid
        if emulator:
            # if not emulator_run:
            #     file = f"{emulator_file_path}emulator_{timestamp}_{receptor_lon}_{receptor_lat}.nc"
            #     foot_data = nc.Dataset(file)
            #     foot = np.float32(np.array(foot_data['foot']))
            #     if mode == 'integrated':
            #         H_valid[idx, index*m:(index+1)*m] = flatten_2d_column(foot)
            #     elif mode == 'integrated_average':
            #         # Validation (not used in the modeling)
            #         foot_step_avg = foot/back_hours
            #         H_check_valid[idx, index*m:(index+1)*m] = flatten_2d_column(foot)
    
            #         # Used in modeling
            #         resolved_time_list = date_range[date_range<=time_foot][-(back_hours+1):]
            #         for jdx, time_hour in enumerate(resolved_time_list):
            #             m_index = time_dict[time_hour]
            #             H_valid[idx, m_index*m:(m_index+1)*m] = flatten_2d_column(foot_step_avg)
            #         pass
            #     elif mode == 'integrated_decayed':
            #         weight_list = create_weights()
            #         # Validation (not used in the modeling)
            #         H_check_valid[idx, index*m:(index+1)*m] = flatten_2d_column(foot)
    
            #         # Used in the modeling
            #         resolved_time_list = date_range[date_range<=time_foot][-(back_hours+1):]
            #         for jdx, time_hour in enumerate(resolved_time_list):
            #             m_index = time_dict[time_hour]
            #             H_valid[idx, m_index*m:(m_index+1)*m] = flatten_2d_column(foot*weight_list[jdx])
                pass
        else:
            if mode == 'integrated':
                foot = np.float32(np.nansum(np.array(foot_data['foot']), axis=0))[clat_index-200:clat_index+200, clon_index-200:clon_index+200]
                H_valid[idx, index*m:(index+1)*m] = flatten_2d_column(foot)
            elif mode == 'integrated_average':
                foot = np.float32(np.average(np.array(foot_data['foot']), axis=0))[clat_index-200:clat_index+200, clon_index-200:clon_index+200]
                # To verify the code (code testing): foot_check and H_check
                foot_check = np.float32(np.nansum(np.array(foot_data['foot']), axis=0))[clat_index-200:clat_index+200, clon_index-200:clon_index+200]
                H_check_valid[idx, index*m:(index+1)*m] = flatten_2d_column(foot_check)
    
                # resolved_time_list = date_range[date_range<=time_foot][-foot.shape[0]:]
                resolved_time_list = date_range[date_range<=time_foot][-(back_hours+1):]
                for jdx, time_hour in enumerate(resolved_time_list):
                    m_index = time_dict[time_hour]
                    H_valid[idx, m_index*m:(m_index+1)*m] = flatten_2d_column(foot)
            elif mode == 'integrated_decayed':
                # declare the weights
                weight_list = create_weights()
                foot = np.float32(np.nansum(np.array(foot_data['foot']), axis=0))[clat_index-200:clat_index+200, clon_index-200:clon_index+200] #compressed footprints
                H_check_valid[idx, index*m:(index+1)*m] = flatten_2d_column(foot)
                resolved_time_list = date_range[date_range<=time_foot][-(back_hours+1):]
                for jdx, time_hour in enumerate(resolved_time_list):
                    m_index = time_dict[time_hour]
                    H_valid[idx, m_index*m:(m_index+1)*m] = flatten_2d_column(foot*weight_list[jdx])
                # import pdb; pdb.set_trace()
            elif mode == 'resolved':
                foot = np.float32(np.array(foot_data['foot']))[:, clat_index-200:clat_index+200, clon_index-200:clon_index+200]
                resolved_time_list = date_range[date_range<=time_foot][-foot.shape[0]:]
                for jdx, time_hour in enumerate(resolved_time_list):
                    m_index = time_dict[time_hour]
                    H_valid[idx, m_index*m:(m_index+1)*m] = flatten_2d_column(foot[jdx, :, :])

        # self.H_valid = H_valid.copy()
        # self.Y_valid = Y_valid.copy()
        # self.validation_dict = validation_dict.copy()
        # self.BKG_valid = BKG_valid.copy()
        # self.H_check_valid = H_check_valid.copy()

    def fill_obs_parallel(self, domain_foot_files, domain_foot_files_validation):
        """
            Parallel implementation of the fill_obs_data function
    
            Arguments:
    
            returns:
                
        """
        print("Reading observation data ....")
        OUTPUT = Parallel(n_jobs=1, verbose=0, backend='threading')(delayed(self.fill_obs_data)(idx, foot_file, mode=mode) for idx, foot_file in tqdm(enumerate(domain_foot_files)))
        # for idx, foot_file in tqdm(enumerate(domain_foot_files)):
        #     self.fill_obs_data(idx, foot_file, mode=mode)
        OUTPUT_valid = Parallel(n_jobs=1, verbose=0, backend='threading')(delayed(self.fill_obs_validation)(idx, foot_file, mode=mode) for idx, foot_file in tqdm(enumerate(domain_foot_files_validation)))
        # import pdb; pdb.set_trace()
        # for idx, foot_file in tqdm(enumerate(domain_foot_files_validation)):
        #     print(foot_file, "check")
        #     self.fill_obs_validation(idx, foot_file, mode=mode)
        if emulator:
            if emulator_run:
                print("Running emulator to generate footprints ....")
                receptor_batch = []
                index_list = []
                for receptor in OUTPUT:
                    receptor_batch.append(receptor[:-1])  # excluding idx
                    index_list.append(idx)
                    runFootEmulator(receptor_batch, index_list)
    
            else:
                print("Reading footprints from stored emulator footprints .....")
                # print(OUTPUT)
                emulator_OUTPUT = Parallel(n_jobs=1, verbose=0, backend='threading')(delayed(readFootEmulator)(receptor, emulator_run=emulator_run) for receptor in tqdm(OUTPUT))
                # emulator_OUTPUT_valid = Parallel(n_jobs=-1, verbose=0, backend='threading')(delayed(readFootEmulator)(receptor, emulator_run=emulator_run) for receptor in tqdm(OUTPUT))
        return


    def load_prior_emissions(self, emission_filtered_files):
        """
            Read the prior emission data
    
            Arguments:
    
            returns:
                
        """
        X = self.X
        print("Loading emission data ....")
        for idx, ems_file in tqdm(enumerate(emission_filtered_files)):
            [_, year, month, day, hour] = parse_ems_info(ems_file)
            year = int(year)
            month = int(month)
            day = int(day)
            hour = int(hour)
            timestamp = datetime.datetime(year, month, day, hour)
            ems_data = np.float32(np.array(nc.Dataset(ems_file)['flx_total']))[clat_index-200:clat_index+200, clon_index-200:clon_index+200]
            ems_flattened = flatten_2d_column(ems_data)
            index = time_dict[timestamp]
            X[index*m:(index+1)*m, 0] = ems_flattened
        # self.X = X.copy()

    def fill_observation_covariance_matrix(self):
        """
            Compute the observational covariance matrix
    
            Arguments:
    
            returns:
                
        """
        
        So = self.So
        observation_dict = self.observation_dict.copy()
        
        # So = np.zeros((So_d.shape[0], So_d.shape[0]), dtype=np.float32)
        print("Forming observation covariance matrix ....")
        nObs = self.So_d.shape[0]
        
        # Adding diagonal terms
        for i in range(nObs):
            So[i, i] = self.So_d[i, 0]
    
        # Adding off-diagonal terms
        for i in tqdm(range(nObs)):
            time_val_i = observation_dict[i]['time']
            coord_i = (observation_dict[i]['lat'], observation_dict[i]['lon'])
            for j in range(i+1, nObs):
                coord_j = (observation_dict[j]['lat'], observation_dict[j]['lon'])
                time_val = (observation_dict[j]['time']-time_val_i).seconds/3600
                dist_val = get_len(coord_i, coord_j)
                time_decay = np.exp(-abs(time_val)/tau_time)
                dist_decay = np.exp(-abs(dist_val)/tau_space)
                sig_val = time_decay*dist_decay*np.sqrt(self.So_d[i]*self.So_d[j])
                if time_decay*dist_decay > lowBound:
                    So[i, j] = sig_val
                    So[j, i] = sig_val
        So = csc_matrix(So)
        self.So = So.copy()


class CalibrationCheckPrior():
    def __init__(self, H, X, Y, So, observation_dict, BKG):
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
        self.BKG = BKG
        
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

        # self.Sa_xy = self.form_spatial_covariance()
        self.Sa_xy = self.form_spatial_covariance(X) # csr array or csc matrix?
        
        self.So = So # csc matrix
        self.X_hat = None

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


mode = 'resolved'
test = True
emulator = False

back_hours = 36
full_prior = True
sparse = True
device = 'cpu'
hq_parallel = True
cross_validation = True
cross_validation_fraction = 0.15

# Directories and files
footprint_directory = "/home/disk/hermes/data/footprints/BEACO2N/obs/"
# emulator_file_path = "/home/disk/hermes/taihe/footnet_test/nikhils_config/footprints/"
emission_directory = "/home/disk/hermes/data/emissions/BEACO2N/"
# emulator_model_file = '/data/Unet_checkpt_0.58_mixed.h5'

# if emulator:
# 	output_directory = f"/home/disk/hermes/nd349/data/inversion/posterior/BEACON/keras_non_distance_L1/{mode}/"
# else:
	# output_directory = f"/home/disk/hermes/nd349/data/inversion/posterior/BEACON/STILT_posterior_BEACON/{mode}/"

# if not os.path.exists(output_directory):
# 	raise Exception(f"The output_directory: {output_directory} does not exist ....")
# else:
# 	pass # You can also make the output directory here if you want!
# print(f"Output directory: {output_directory}:")

model_error = { 0:2, 1:4, 2:6, 3:8, 4:5, 5:4, 6:3, 7:3, 8:3, 9:3, 10:3, 11:3, 12:3, \
13:3, 14:4, 15:5, 16:8, 17:6, 18:4, 19:2, 20:1, 21:1, 22:1, 23:1}

# Uncertainties
ems_uncert = 50/100
minUncert = 1.0
tau_day = 1
tau_hr = 5
tau_len = 5
tau_time = 1 # hour
tau_space = 2 # km
mVal = minUncert/ems_uncert
lowBound = 1e-5
obsFreq    = 60.0    # Observations per hour
fsigma=0.5

### Different grids to use
# Full grid used for footprints
full_xLim = [ -125.0, -120.0 ]
full_yLim = [   36.0,   40.0 ]
big_xLim,big_yLim = full_xLim,full_yLim
num_lats = 481
num_lons = 601

# Grid
# lats = np.linspace(full_yLim[0], full_yLim[1], num_lats, dtype=np.float32)
# lons = np.linspace(full_xLim[0], full_xLim[1], num_lons, dtype=np.float32)

orig_lats = np.linspace(full_yLim[0], full_yLim[1], num_lats)
orig_lons = np.linspace(full_xLim[0], full_xLim[1], num_lons)

clon_index = int(orig_lons.shape[0]/2)
clat_index = int(orig_lats.shape[0]/2)
clon = orig_lats[clon_index]
clat = orig_lons[clat_index]
lats = orig_lats[clat_index-200:clat_index+200]
lons = orig_lons[clon_index-200:clon_index+200]

nrow = 400
ncol = 400
m = nrow*ncol

# Medium sized grid
medium_xLim = [-123.60,-121.60]
medium_yLim = [  36.80,  38.60]
med_xLim,med_yLim = medium_xLim,medium_yLim
# Bay Area domain (smallest grid)
BayArea_xLim = [-123.10,-121.80]
BayArea_yLim = [  37.35,  38.40]
small_xLim,small_yLim = BayArea_xLim,BayArea_yLim
# Inversion grid to use
Inv_lonLim = small_xLim
Inv_latLim = small_yLim

footprint_files = get_files(footprint_directory)
emission_files = get_files(emission_directory)
emission_files.sort()
footprint_files.sort()
print(len(footprint_files), len(emission_files))

# Time domain
if test:
	print("job field is empty:", sys.argv)
	start_time = datetime.datetime(2020, 2, 10, 0, 0)
	end_time = datetime.datetime(2020, 2, 11, 0, 0)

else:
	start_time = sys.argv[0] #yyyymmddhh (str)
	end_time = sys.argv[1] #yyyymmddhh (str)
	if len(start_time) !=10 or len(end_time)!=10:
		raise Exception(f"The expected format for the start and end date is yyyymmddhh but {start_time} {end_time} is given")
	else:
		start_time = datetime.datetime(int(start_time[0:4]), int(start_time[4:6]), int(start_time[6:8]), int(start_time[8:10]))
		end_time = datetime.datetime(int(end_time[0:4]), int(end_time[4:6]), int(end_time[6:8]), int(end_time[8:10]))

if mode == 'resolved':
	m_start = start_time-datetime.timedelta(hours=back_hours)
	m_end = end_time+datetime.timedelta(hours=back_hours)
	m_end = m_end-datetime.timedelta(hours=1)
elif mode == 'integrated' or mode == 'integrated_average' or mode == 'integrated_decayed': # Check with Alex
	m_start = start_time-datetime.timedelta(hours=back_hours)
	m_end = end_time+datetime.timedelta(hours=back_hours)
	m_end = m_end-datetime.timedelta(hours=1)


date_range = pd.date_range(start=m_start, end=m_end, freq='1h')

time_dict = {}
for idx, value in enumerate(date_range):
    time_dict[value] = idx

data = read_data(m_start, m_end, cross_validation)

cal = CalibrationCheckPrior(data.H, data.X, data.Y, data.So, data.observation_dict, data.BKG)
squared_error1 = np.array(cal.Y - np.array(np.dot(cal.H_array, cal.X_pri_array)))**2
print("Squared error:", squared_error1)
print("Sum of Squared error:", np.sum(squared_error1))
HP = HQ(cal.H_array, cal.Sa_t, cal.Sa_xy, parallel=True)
print(HP)

HPHT = HQHT(HP, cal.H_array, cal.Sa_t, cal.Sa_xy)
print(HPHT)

print(np.sum(HPHT + cal.So))
