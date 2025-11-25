import sys
# sys.path.append("../../PythonScripts")
# from pipeline_functions import build_ply_files
import os
import pandas as pd
import numpy as np
from plyfile import PlyData, PlyElement


ROOT = """/home/sspiegel/CapstoneData/Paris/Toronto_3D/pickleFiles/2025_11_04T18_57_L001_cylinder_r_0_1_grid_0_02_features.npz"""

das = np.load(ROOT)["array2"]

xyz = np.load(ROOT)["array1"]

cls = np.load(ROOT)["array3"]

cls[cls==2] = 1

cls[cls > 1] -= 1

cols = ["EigenSum","omnivariance","entropy","linearity","planarity","sphericity","curvature","verticality1","verticality2","count"]

print(xyz.shape)
print(das.shape)
print(cls.shape)

allCols = ['X', 'Y','Z'] + cols + ['label']

allAtrs = np.hstack((xyz,das, cls.reshape(-1, 1)))

total_dataframe = pd.DataFrame(allAtrs, columns=allCols)
total_dataframe["label"] = total_dataframe["label"].astype(int)
total_dataframe["count"] = total_dataframe["count"].astype(int)

tpsOut = []
for idx, tpe in total_dataframe.dtypes.to_dict().items():
    if tpe == 'int64':
        tpsOut.append((idx, 'i4'))
    elif tpe == 'float64':
        tpsOut.append((idx, 'f8'))


vertex_data = np.empty(allAtrs.shape[0], dtype=tpsOut)


for t in tpsOut:
    vertex_data[t[0]] = total_dataframe[t[0]].values

el = PlyElement.describe(vertex_data, 'vertex')

PlyData([el], text=False).write(f"""/home/sspiegel/CapstoneData/Paris/Toronto_3D/PC_with_features/L001_cylinder_features_radius_01.ply""")


    