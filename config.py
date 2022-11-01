# -*- coding: utf-8 -*-
# @Author: nikhildadheech
# @Date:   2022-08-23 12:37:29
# @Last Modified by:   nikhildadheech
# @Last Modified time: 2022-10-30 23:06:58


import datetime # ; import time
import numpy as np
import pandas as pd
import sys

sys.argv = sys.argv[1:]
print("sys.argv:", sys.argv)
mode = 'integrated'

slurm_run = False
if slurm_run:
	job = str(sys.argv[2])
else:
	job = ''
location = job.upper()
back_hours = 24
diag_prior = False
full_prior = True
sparse = True
device = 'cpu'

# Directories and files
footprint_directory = "/home/disk/hermes/data/footprints/BEACO2N/obs/"
emission_directory = "/home/disk/hermes/data/emissions/BEACO2N/"
output_directory = "/home/disk/hermes/nd349/data/inversion/integrated_BEACON/"



# Model error
model_error = { 0:3, 1:3, 2:3, 3:3, 4:3, 5:3, 6:4, 7:5, 8:8, 9:6, 10:4, 11:2, 12:1, \
13:1, 14:1, 15:1, 16:2, 17:4, 18:6, 19:8, 20:6, 21:4, 22:3, 23:3}

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
nrow = num_lats
num_lons = 601
ncol = num_lons
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


# Grid
lats = np.linspace(full_yLim[0], full_yLim[1], num_lats, dtype=np.float32)
lons = np.linspace(full_xLim[0], full_xLim[1], num_lons, dtype=np.float32)


# Time domain
if not slurm_run:
	print("job field is empty:", sys.argv)
	start_time = datetime.datetime(2018, 1, 4, 0, 0)
	end_time = datetime.datetime(2018, 1, 5, 0, 0)

if slurm_run:
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


output_posterior_file = f"/home/disk/p/nd349/nikhil.dadheech/pointSources/Inversion/BEACON_Inv_python/data/inversion/\
posterior{str(m_start)}_{str(m_end)}_{mode}_{device}.nc".replace(" ", "_")




