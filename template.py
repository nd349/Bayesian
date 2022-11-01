# -*- coding: utf-8 -*-
# @Author: nikhildadheech
# @Date:   2022-07-25 14:58:38
# @Last Modified by:   nikhildadheech
# @Last Modified time: 2022-10-07 11:39:57

# from diagPrior.time_resolved_diag_prior import InversionDiagPrior

from Utils.getData import *
from config import *
from fullCovariance.InversionFullCovariance import InversionFullPrior
import pickle

invert = InversionFullPrior(H, X, Y, So)
X_hat = invert.invert()
print("Size of the posterior solution:", X_hat.shape)
invert.save_solution()

# with open(output_posterior_file, "wb") as write_file:
# 	pickle.dump(X_hat, write_file)

