import open3d as o3d
import open3d.visualization as vis
import numpy as np
from sklearn.neighbors import KDTree

X = 20*np.random.rand(10000, 3)

tree = KDTree(X)

distMat = tree.query_radius(X, 3)

idx = np.random.randint(0, X.shape[0])


xyzPoint = o3d.geometry.PointCloud()

xyz = o3d.geometry.PointCloud()


xyz.points = o3d.utility.Vector3dVector(X)
xyz.paint_uniform_color([0,0,1])

xyzPoint.points = o3d.utility.Vector3dVector(X[idx].reshape(1,-1))



xyzPoint.paint_uniform_color([1.0, 0.0, 0.0])

neighs = o3d.geometry.PointCloud()
neighs.points = o3d.utility.Vector3dVector(X[distMat[idx]])

neighs.paint_uniform_color([0.0, 1.0, 0.0])



sphere = o3d.geometry.TriangleMesh.create_sphere(3.0)
sphere.compute_vertex_normals()
sphere.translate(X[idx])
mat_sphere = vis.rendering.MaterialRecord()
mat_sphere.shader = 'defaultLitTransparency'
mat_sphere.base_color = [0.467, 0.467, 0.467, 0.2] 

gems = [{'name' : 'allPoints', 'geometry':xyz} ,
{'name' : 'neighbors', 'geometry' : neighs},
{'name' : 'basePoint', 'geometry' : xyzPoint},
{'name': 'sphere', 'geometry': sphere, 'material': mat_sphere}]


vis.draw(gems)
