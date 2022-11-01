# -*- coding: utf-8 -*-
# @Author: nikhildadheech
# @Date:   2022-08-23 12:37:29
# @Last Modified by:   nikhildadheech
# @Last Modified time: 2022-10-03 15:26:47

import os
import datetime # ; import time
import numpy as np
import pandas as pd

location = 'BEACON'
start = datetime.datetime(2020, 3, 22, 0, 0)
end = datetime.datetime(2020, 5, 3, 0, 0)
hours_interval = 24 # hours
difference = (end-start).total_seconds()/3600 # hours
iteration = int(difference/hours_interval)

def create_submission_bash(jobname):

    with open('job.sh', 'w') as file:
        file.writelines("#!/bin/bash\n")
        file.writelines("\n")
        file.writelines("#SBATCH\n")
        file.writelines(f"#SBATCH --job-name={jobname}\n")
        file.writelines("#SBATCH -N 1      # nodes requested\n")
        file.writelines("#SBATCH -n 1      # tasks requested\n")
        file.writelines("#SBATCH -c 32      # cores requested\n")
        file.writelines("#SBATCH --mem=102400  # memory in Mb\n")
        file.writelines(f"#SBATCH -o logs/slurm_output_{jobname}_job%j.out  # send stdout to outfile\n")
        file.writelines(f"#SBATCH -e logs/slurm_errfile_{jobname}_job%j.out  # send stderr to errfile\n")
        file.writelines("#SBATCH -t 96:00:00  # time requested in hour:minute:second\n")
        file.writelines("\n\n\n")
        file.writelines("cd /home/disk/p/nd349/nikhil.dadheech/pointSources/Inversion/BEACON_Inv_python\n")
        file.writelines("source /home/disk/p/nd349/anaconda3/etc/profile.d/conda.sh\n")
        file.writelines("\n\n\n")
        file.writelines("conda activate stilt\n")
        file.writelines("time python template.py $1 $2 $3")
    file.close()

for idx in range(iteration):
    batch_start = start+datetime.timedelta(hours=hours_interval*(idx))
    batch_end = start+datetime.timedelta(hours=hours_interval*(idx+1))

    batch_start_year = str(batch_start.year)
    batch_start_month = str(batch_start.month)
    batch_start_day = str(batch_start.day)
    batch_start_hour = str(batch_start.hour)

    batch_end_year = str(batch_end.year)
    batch_end_month = str(batch_end.month)
    batch_end_day = str(batch_end.day)
    batch_end_hour = str(batch_end.hour)

    if len(batch_start_month) == 1:
        batch_start_month = '0' + batch_start_month
    if len(batch_start_day) == 1:
            batch_start_day = '0' + batch_start_day
    if len(batch_start_hour) == 1:
            batch_start_hour = '0' + batch_start_hour
    if len(batch_end_month) == 1:
        batch_end_month = '0' + batch_end_month
    if len(batch_end_day) == 1:
            batch_end_day = '0' + batch_end_day
    if len(batch_end_hour) == 1:
            batch_end_hour = '0' + batch_end_hour

    batch_start = f"{batch_start_year}{batch_start_month}{batch_start_day}{batch_start_hour}"
    batch_end = f"{batch_end_year}{batch_end_month}{batch_end_day}{batch_end_hour}"

    create_submission_bash(f"{batch_start}")

    print (f"sbatch job.sh {batch_start} {batch_end} {location}")
    os.system(f"sbatch job.sh {batch_start} {batch_end} {location}")
    # os command to submit the job

