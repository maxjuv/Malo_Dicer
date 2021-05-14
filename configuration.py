# !/Users/maximejuventin/anaconda3/bin/python
import os
import sys
# print(os.system('which python'))
# print(os.system('python --version'))
# os.system('/Users/maximejuventin/anaconda3/bin/python')
# print(os.system('which python'))
import getpass

import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime

# work_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__))) + '/'
work_dir = '/Users/maximejuventin/Desktop/DICER/'
print(work_dir)
data_dir = work_dir + '/data/'
excel_dir = work_dir + '/excels/'
if getpass.getuser() in ('samuel.garcia', 'maxime.juventin') and  sys.platform.startswith('linux'):
    precompute_dir = '/mnt/data/CAmatPion201801_intra_respi_Maxime/amalo/precompute/'
else :
    precompute_dir = work_dir + '/precompute/'
    # precompute_dir = 'Z:/precompute_edf/'


if __name__ == '__main__':
    print(work_dir)
