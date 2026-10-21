import os
import numpy as np
import pandas as pd

from mne_bids import BIDSPath, read_raw_bids
ass=np.load("dataset/subject_P200/X.npy").shape
ssss=np.load("dataset/subject_P201/X.npy").shape

print(ass)
print(ssss)