import os
from os import listdir
import datetime # ; import time
import numpy as np
import pandas as pd
import sys

mode = 'integrated_decayed'
experiment = 'emulator'+mode

location = 'BEACON'+experiment
start = datetime.datetime(2020, 2, 2, 0, 0)
end = datetime.datetime(2020, 5, 2, 0, 0)

date_range = pd.date_range(start=start, end=end, freq='24h')

timestamps = [datetime.datetime.strftime(val, '%Y%m%d%H')[:8] for val in date_range]

timestamps = set([f"{val[:4]}x{val[4:6]}x{val[6:]}" for val in timestamps])
# print(timestamps, len(timestamps))

def get_files(directory, extension=""):
    """
        Get the relevant files from a given path

        Arguments:
            directory: <str>
            extension: <str>
        returns:
            files: <list>
    """
    if extension:
        files = [f for f in listdir(directory) if f[-len(extension):] == extension]
    else:
        files = [f for f in listdir(directory)]
    files = [directory+file for file in files]
    return files

files = get_files(f"/home/disk/hermes/nd349/data/inversion/posterior/BEACON/BKG_FootNet_base_no_dist_MSE_withoutAD/{mode}/", extension=".ncdf")
files = set([val.split("_")[-1][:-8] for val in files])
# print(files)

remaining_timestamps = list(timestamps - files.intersection(timestamps))
remaining_timestamps = [val.replace("x", "")+"00" for val in remaining_timestamps]
remaining_timestamps = [(datetime.datetime.strptime(val, '%Y%m%d%H'), datetime.datetime.strptime(val, '%Y%m%d%H') + datetime.timedelta(hours=24)) for val in remaining_timestamps]
remaining_timestamps = [(datetime.datetime.strftime(val[0], '%Y%m%d%H'), datetime.datetime.strftime(val[0] + datetime.timedelta(hours=24), '%Y%m%d%H')) for val in remaining_timestamps]
remaining_timestamps.sort(key=lambda x:x[0])
print("Remaining_timestamps", len(remaining_timestamps))


run_model = True
slurm = False


nodelist = ['h5', 'h6', 'h7', 'h9', ]
use_nodelist = False

def create_submission_bash(date, term, node=''):
    """
        Update the submission script job.sh

        Arguments:
            date: <str>
            term: <str>
        returns:
            None
    """

    with open('job.sh', 'w') as file:
        jobname = date+term
        file.writelines("#!/bin/bash\n")
        file.writelines("\n")
        file.writelines("#SBATCH\n")
        file.writelines(f"#SBATCH --job-name={jobname}\n")
        file.writelines("#SBATCH -N 1      # nodes requested\n")
        file.writelines("#SBATCH -n 1      # tasks requested\n")
        file.writelines("#SBATCH -c 32      # cores requested\n")
        if node:
            file.writelines(f"#SBATCH --nodelist={node}      # nodes requested\n")
        file.writelines("#SBATCH --mem=102400  # memory in Mb\n")
        file.writelines(f"#SBATCH -o /home/disk/hermes/nd349/data/inversion/runs/logs/{mode}/{jobname}.out  # send stdout to outfile\n")
        # file.writelines(f"#SBATCH -e /home/disk/hermes/nd349/data/inversion/runs/logs/slurm_errfile_{jobname}.out  # send stderr to errfile\n")
        file.writelines("#SBATCH -t 96:00:00  # time requested in hour:minute:second\n")
        file.writelines("\n\n\n")
        file.writelines("cd /home/disk/p/nd349/nikhil.dadheech/pointSources/Inversion/InversionEmulator/BEACON_emulator400400/\n")
        file.writelines("source /home/disk/p/nd349/anaconda3/etc/profile.d/conda.sh\n")
        file.writelines("\n\n\n")
        file.writelines("conda activate h2\n")
        file.writelines("time python template.py $1 $2 $3 $4")
    file.close()
    

for idx, val in enumerate(remaining_timestamps):
    print(idx, val)
    batch_start = val[0]
    batch_end = val[1]
    node = nodelist[idx%len(nodelist)]
    if run_model:
        if slurm:
            if use_nodelist:
                create_submission_bash(f"{batch_start}", f"{experiment}", node)
            else:
                create_submission_bash(f"{batch_start}", f"{experiment}", node='')

            print (f"sbatch job.sh {batch_start} {batch_end} {location} {mode}")
            os.system(f"sbatch job.sh {batch_start} {batch_end} {location} {mode}")
        
        else:
            print("Running the model here ...")
            os.system(f"python template.py {batch_start} {batch_end} {location} {mode}")
