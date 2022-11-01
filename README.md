# BayesianInferenceInversion\n

Main file: template.py\n
Config file: config.py\n

Following steps to start running the inversion code:\n
(a) First change parameters in the config.py\n
(b) Generate spatial correlation matrix\n
(c) Update the read data script as per your input data format and make sure the data (H, X, Y, and covariance matrices) are in the correct format\n
(d) You can use slurm_run/submit_slurm.py file to submit multiple jobs in parallel across multiple nodes using slurm job scheduler\n

For more questions contact: nd349@uw.edu
