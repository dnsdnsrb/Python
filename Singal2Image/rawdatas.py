import numpy as np
import os
from nptdms import TdmsFile
import glob
import pandas as pd
import datetime
from matplotlib import pyplot as plt
from PIL import Image

base_dir = os.path.dirname(os.path.realpath(__file__))
file_paths = glob.glob(os.path.join(base_dir, 'RawDatas', '*.tdms'))
print(file_paths)
# file_paths = file_paths[:1]

df_list = []
for file_path in file_paths:
    with TdmsFile.read(file_path) as tdms_file:
        df = tdms_file.as_dataframe(scaled_data=False)
        df_list.append(df)

df = pd.concat(df_list)
df.to_numpy()
# with open('Rawdatas.npy', 'wb') as f:
#     np.save(f, df)


