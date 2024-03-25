# -*- coding: utf-8 -*-
# @Author: nikhildadheech
# @Date:   2022-09-28 14:20:22
# @Last Modified by:   nikhildadheech
# @Last Modified time: 2022-09-28 16:33:33


#!/usr/bin/env python
#SBATCH --job-name=Python

import sys
import os
import time

print(sys.argv)

if len(sys.argv) != 2:
	print('Usage: %s MAXIMUM' % (os.path.basename(sys.argv[0])))
	sys.exit(1)

maximum = int(sys.argv[1])

n1 = 1
n2 = 1

while n2 <= maximum:
  n1, n2 = n2, n1 + n2
  # time.sleep(5)

print('The greatest Fibonacci number up to %d is %d' % (maximum, n1))

