import numpy as np
from stl import mesh
# from mpl_toolkits import mplot3d
# from matplotlib import pyplot
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

my_mesh = mesh.Mesh.from_file('CRANIAL HEADS_Head_1_001_centered.stl')

print(my_mesh.points[0])
print(my_mesh.vectors[0])

all_pts = []
for vector in my_mesh.vectors:
  all_pts.extend(vector / 1000)

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

"""
figure = pyplot.figure()
axes = mplot3d.Axes3D(figure)

axes.add_collection3d(mplot3d.art3d.Poly3DCollection(my_mesh.vectors))

scale = my_mesh.points.flatten(-1)
axes.auto_scale_xyz(scale, scale, scale)

pyplot.show()
"""
