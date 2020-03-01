# test_stl.py
# author: Austin Shin

import matplotlib.pyplot as plt
import numpy as np

from mpl_toolkits.mplot3d import Axes3D
from stl import mesh

my_mesh = mesh.Mesh.from_file('data/CRANIAL HEADS_Head_1_001_centered.stl')

all_pts = []
for vector in my_mesh.vectors:
  all_pts.extend(vector / 1000) # convert to meters

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
pts = np.array(all_pts)
# ax.scatter(pts[:,0], pts[:,1], pts[:,2])

num = pts.size / 3
indices = np.random.choice(num, 1200)
# print(pts[indices,:])
sub_pts = pts[indices,:]
ax.scatter(sub_pts[:,0], sub_pts[:,1], sub_pts[:,2])

plt.show()
