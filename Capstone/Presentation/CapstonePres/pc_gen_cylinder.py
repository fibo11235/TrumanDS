import open3d as o3d
import open3d.visualization as vis
import numpy as np
from sklearn.neighbors import KDTree

####################### Custom Cylinder
class Cylinder(object):
    """
    Cylinder object for neighborhood search in point clouds.

    This class defines a cylinder based on a search radius R, where:
        - self.r: radius of the cylinder (computed as R / sqrt(2))
        - self.h: height of the cylinder (computed as 2*R / sqrt(2))
    """

    def __init__(self, R: float) -> None:
        """
        Initialize a Cylinder object with a given search radius.

        Args:
            R (float): Search radius (typically the radius of the sphere).

        Sets:
            self.r (float): Radius of the cylinder.
            self.h (float): Height of the cylinder.
        """
        self.r = R / np.sqrt(2)
        self.h = 2*R / np.sqrt(2)

        
    def computePoints(self, BasePC : np.array, CandidatePoints : np.array, distMat: np.array) -> np.array:

        """
            Get points that fit within the cylinder object centered at a point
            input:
                    BasePC : Center point of neighborhood
                    CandidatePoints : downsampled point cloud that is within R distance of BasePoint
                    distMat : array of indicies of points that are within R distance of Base Point
            output:
                    CandidateIDs : array of indicies that are within the cylinder centered at BasePoint
        """

        if BasePC.ndim != 2:
            BasePC = np.squeeze(BasePC).reshape(1, -1)
        
        BasePointTop = BasePC + np.array([[0.,0.,self.h/2]])
        BasePointBottom = BasePC - np.array([[0.,0.,self.h/2]])

        CandidatePointsIDS = np.hstack((CandidatePoints, distMat.reshape(-1,1))) # Keep track of the indexes
        

        bb = CandidatePointsIDS[(CandidatePointsIDS[:,2] <= BasePointTop[:,2]) & # Get points with z coordinates less than top point
        (CandidatePointsIDS[:,2] >= BasePointBottom[:,2]) &  # Get points with z coordinates less than bottom point
        (np.sum(((CandidatePointsIDS[:,0] - BasePC[:,0])**2, (CandidatePointsIDS[:,1] - BasePC[:,1])**2), axis = 0) <= self.r**2)] # Get points that are within r distance

        return bb[:,-1].astype(int)
############################################################################


radius = 3.0

cyl = Cylinder(radius)

X = 20*np.random.rand(10000, 3)

tree = KDTree(X)


distMat = tree.query_radius(X, radius)




idx = np.random.randint(0, X.shape[0])


xyzPoint = o3d.geometry.PointCloud()

xyz = o3d.geometry.PointCloud()


xyz.points = o3d.utility.Vector3dVector(X)
xyz.paint_uniform_color([0,0,1])

xyzPoint.points = o3d.utility.Vector3dVector(X[idx].reshape(1,-1))



xyzPoint.paint_uniform_color([1.0, 0.0, 0.0])

neighs = o3d.geometry.PointCloud()

neighsSphere = o3d.geometry.PointCloud()

### Replace the neighbors
print(distMat[idx])
distMatFix = cyl.computePoints(X[idx], X[distMat[idx]], distMat[idx])

neighs.points = o3d.utility.Vector3dVector(X[distMatFix])

neighs.paint_uniform_color([0.0, 1.0, 0.0])

############ Keep sphere

neighsSphere.points = o3d.utility.Vector3dVector(X[distMat[idx]])

neighsSphere.paint_uniform_color([0.0, 1.0, 1.0])




sphere = o3d.geometry.TriangleMesh.create_sphere(radius)

cylinder = o3d.geometry.TriangleMesh.create_cylinder(cyl.r, cyl.h) # Create cininder
cylinder.compute_vertex_normals()
cylinder.translate(X[idx])
mat_cylinder = vis.rendering.MaterialRecord()
mat_cylinder.shader = 'defaultLitTransparency'
mat_cylinder.base_color = [0.467, 0.467, 0.467, 0.6] 

sphere.compute_vertex_normals()
sphere.translate(X[idx])
mat_sphere = vis.rendering.MaterialRecord()
mat_sphere.shader = 'defaultLitTransparency'
mat_sphere.base_color = [0.467, 0.467, 0.467, 0.2] 

gems = [{'name' : 'allPoints', 'geometry':xyz},
{'name' : 'neighborsSpherical', 'geometry' : neighsSphere},
{'name' : 'neighborsCylinder', 'geometry' : neighs},
{'name' : 'basePoint', 'geometry' : xyzPoint},
{'name': 'sphere', 'geometry': sphere, 'material': mat_sphere},
{'name': 'cylinder', 'geometry': cylinder, 'material': mat_cylinder}]


vis.draw(gems)
