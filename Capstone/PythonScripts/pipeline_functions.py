import datetime
import pdal
import json
import numpy as np
import open3d as o3d
from datetime import datetime, timezone
import multiprocessing as mp 
from scipy.spatial import cKDTree   

js = """{
    "pipeline": [
        {
            "type": "readers.ept",
            "filename":"http://usgs-lidar-public.s3.amazonaws.com/MO_WestCentral_2_2018/ept.json"

        },
        {
            "type": "filters.reprojection",
            "out_srs": "EPSG:4326"
        },
        {
            "type" : "filters.reprojection",
            "in_srs" : "EPSG:4326",
            "out_srs" : "EPSG:26915"
        },
        {
            "type": "writers.las"        
        }
    ]
}"""



def import_las(path) -> tuple:
    """
    Read a LAS/LAZ file using PDAL (_pdal).

    Returns a tuple: (points_structured_array, xyz_array_or_None, metadata_dict_or_raw, num_points)
    - points_structured_array: numpy structured array with fields from the LAS file
    - xyz_array_or_none: (N,3) float32/float64 array of X,Y,Z if present, otherwise None
    - metadata_dict_or_raw: parsed metadata dict when possible, otherwise raw metadata string
    - num_points: number of points read (int)
    """


    pipeline_spec = {"pipeline": [{"type": "readers.las", "filename": path}]}
    pipeline = pdal.Pipeline(json.dumps(pipeline_spec))
    num_points = pipeline.execute()
    arrays = pipeline.arrays
    if not arrays:
        return np.empty(0, dtype=np.float32), None, {}, int(num_points)
    pts = arrays[0]

    xyz = None
    xyz = np.vstack([pts["X"], pts["Y"], pts["Z"]]).T

    try:
        metadata = json.loads(pipeline.metadata)
    except Exception:
        metadata = pipeline.metadata

    return pts, xyz, metadata, int(num_points)





def generateLas(js : str, X_bound : list, Y_bound: list, splitter : float, metadata_dump : str = "/home/sspiegel/CapstoneData/metadata") -> None:
   
    """
    Pass a PDAL pipeline to generate multiple LAS Files
    input:
        js: input JSON string to generate point cloud
        X_bound: list of minx, maxx
        Y_bound: list of miny, maxy
    """
    fmt = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")

    cntr = 0
    x_min = X_bound[0]
    x_max = X_bound[1]
    y_min = Y_bound[0]
    y_max = Y_bound[1]  

    x_split = np.arange(x_min, x_max, splitter)
    y_split = np.arange(y_min, y_max, splitter)



    for i in range(len(x_split)-1):
        for j in range(len(y_split)-1):
            

            if (cntr + 1) % 10 == 0:
                print(f"""Processing Xbounds {x_split[i]} - {x_split[i + 1]}, YBounds {y_split[0]} - {y_split[1]}\n\n""")
            bounds = f"([{x_split[i]},{x_split[i+1]}],[{y_split[j]},{y_split[j+1]}])"
            pipeline_spec = json.loads(js)
            pipeline_spec["pipeline"][0]["bounds"] = bounds
            pipeline_spec["pipeline"][-1]["filename"] = f"st_peters_{i}_{j}_{fmt}.laz"
            pipeline = pdal.Pipeline(json.dumps(pipeline_spec))
            num_points = pipeline.execute()
            arrays = pipeline.arrays
            if not arrays:
                continue
            pts = arrays[0]
            metadata = pipeline.metadata
            with open(f"{metadata_dump}/st_peters_{i}_{j}_{fmt}_metadata.json", "w") as f:
                f.write(json.dumps(metadata))
            cntr += 1



def downscale_las(input_las: str, sample_size: float) -> None:
    """
    Downsample point cloud using barycenteric method
    input:
        input_las: path to input las file
        sample_size: size of the voxel to downsample to
    output:
        None
    """

    outs = f"""{str(sample_size).split('.')[0]}_{str(sample_size).split('.')[1]}"""

    output = input_las.split(".")[0] + f"_downsampled_{outs}.las"


    st = {
        "pipeline" :
        [
    {
        "type": "readers.las",
        "filename" : input_las
    },
    {
        "type":"filters.voxelcentroidnearestneighbor",
        "cell": sample_size
    },
    {
        "type":"writers.las",
        "filename": output
    }
        ]
    }

    pipeline = pdal.Pipeline(json.dumps(st))
    outs = pipeline.execute()


def downscale_ply(input_ply: str, sample_size: float) -> None:
    """
    Downsample point cloud using barycenteric method that's in PLY format
    input:
        input_ply: path to input ply file
        sample_size: size of the voxel to downsample to
    output:
        None
    """

    outs = f"""{str(sample_size).split('.')[0]}_{str(sample_size).split('.')[1]}"""

    output = input_ply.split(".")[0] + f"_downsampled_{outs}.ply"


    st = {
        "pipeline" :
        [
    {
        "type": "readers.ply",
        "filename" : input_ply
    },
    {
        "type":"filters.voxelcentroidnearestneighbor",
        "cell": sample_size
    },
    {
        "type":"writers.ply",
        "filename": output
    }
        ]
    }

    pipeline = pdal.Pipeline(json.dumps(st))
    outs = pipeline.execute()



def get_scales(r_0 : float = 0.1, S : int = 8, rho : float = 5, gamma : float = 2) -> list:
    """
    Return list of grid sizes and radii for multiscale feature extraction
    input: 
        r_0: base scale
        S: number of scales
        rho: scaling factor
        gamma: scaling factor
    output:
        list of tuples (radius of search, grid size of downscaled point cloud)
    """
    scales = []
    for s in range(S):
        radii = r_0 * gamma ** s
        grid = radii / rho
        scales.append((radii, grid))
    return scales

resList = []

featList = []
covList = []
eigList = []
featList = []


def getFeatures(point_data : np.array,test_pc : o3d.geometry.PointCloud,tree : o3d.geometry.KDTreeFlann,r: float) -> dict:
# def getFeatures(point_data : np.array,tree : cKDTree,r: float) -> dict:

    """
    Return dictionary of geometric features for a given point and radius
    input: 
        point_data: (3,1) array of point coordinates
        test_pc : open3d point cloud object of downsampled point cloud
        tree : KDTree object of downsampled point cloud
        r : radius of search
    output:
        dictionary of geometric features
    """
    e_z = np.array([[0.,0.,1]]) # unit vector in z direction
    srch = tree.search_radius_vector_3d(point_data, r)[1]
    srch = np.asarray(srch)
    done_points = np.asarray(test_pc.points)[srch]
    N = done_points.shape[0]
    if done_points.shape[0] < 5:
        feat = np.array((0., 0., 0., 0., 0., 0., 0., 0., 0., N))
        cov = np.zeros((3,3))
        sorted_eigs = np.zeros(3)
        sorted_vecs = np.zeros((3,3))
    else:
        avg =  done_points - np.mean(done_points, axis=0)
        cov  = np.cov(avg.T, bias=True)
        # covList.append(cov)
        eigs, vecs = np.linalg.eig(cov)
        # eigList.append(eigs)
        sort_indices = np.argsort(eigs)
        sorted_eigs = eigs[sort_indices]
        sorted_vecs = vecs[:, sort_indices]



        sum_of_eigs = sorted_eigs.sum()
        omni = sorted_eigs.prod()**(1/3)
        entro = -1 * (sorted_eigs*np.log((sorted_eigs + 1e-9))).sum()
        lin = (sorted_eigs[0] - sorted_eigs[1]) / (sorted_eigs[0] + 1e-9)
        pln = (sorted_eigs[1] - sorted_eigs[2]) / (sorted_eigs[0] + 1e-9)
        sph = sorted_eigs[2] / (sorted_eigs[0] + 1e-9)
        curv = sorted_eigs[2] / np.sum(sorted_eigs)
        vert_1 = np.squeeze(np.abs((np.pi / 2) - np.arccos(sorted_vecs[0].reshape(1,-1)@e_z.T)))
        vert_2 = np.squeeze(np.abs((np.pi / 2) - np.arccos(sorted_vecs[2].reshape(1,-1)@e_z.T)))


        feat = np.array((sum_of_eigs, omni, entro, lin, pln, sph, curv, vert_1, vert_2, N))

    return {"features" : feat, "covariance_matrix" : cov, "eigenvalues" : sorted_eigs, "eigenvectors" : sorted_vecs}



# def getFeaturesParallel(point_data : np.array,r: float) -> dict:
def getFeaturesParallel(test_pc : np.array, neighs : np.array) -> np.array:

    """
    Return dictionary of geometric features for a given point and radius (using parallel processing)
    input: 
        test_pc :  point cloud object of downsampled point cloud
        neighs : Index of neighbors
    output:
        array of geometric features
    """
    e_z = np.array([[0.,0.,1]]) # unit vector in z direction
    done_points  = test_pc[neighs]
    N = done_points.shape[0]
    if done_points.shape[0] < 5: # Maintain for stability
        feat = np.array((0., 0., 0., 0., 0., 0., 0., 0., 0., N))
        cov = np.zeros((3,3))
        sorted_eigs = np.zeros(3)
        sorted_vecs = np.zeros((3,3))
    else:
        avg =  done_points - np.mean(done_points, axis=0)
        cov  = np.cov(avg.T, bias=True)
        # covList.append(cov)
        eigs, vecs = np.linalg.eig(cov)
        sort_indices = np.argsort(eigs)[::-1]
        sorted_eigs = eigs[sort_indices]
        sorted_vecs = vecs[:, sort_indices]



        sum_of_eigs = sorted_eigs.sum()
        omni = sorted_eigs.prod()**(1/3)
        entro = -1 * (sorted_eigs*np.log((sorted_eigs + 1e-9))).sum()
        lin = (sorted_eigs[0] - sorted_eigs[1]) / (sorted_eigs[0] + 1e-9)
        pln = (sorted_eigs[1] - sorted_eigs[2]) / (sorted_eigs[0] + 1e-9)
        sph = sorted_eigs[2] / (sorted_eigs[0] + 1e-9)
        curv = sorted_eigs[2] / np.sum(sorted_eigs)
        vert_1 = np.squeeze(np.abs((np.pi / 2) - np.arccos(sorted_vecs[0].reshape(1,-1)@e_z.T)))
        vert_2 = np.squeeze(np.abs((np.pi / 2) - np.arccos(sorted_vecs[2].reshape(1,-1)@e_z.T)))


        feat = np.array((sum_of_eigs, omni, entro, lin, pln, sph, curv, vert_1, vert_2, N))

    # return {"features" : feat, "covariance_matrix" : cov, "eigenvalues" : sorted_eigs, "eigenvectors" : sorted_vecs}
    return feat


if __name__ == "__main__":
    X_bound = [-10099875.0,-10083682.0]
    Y_bound = [4685827.0,4696129.0]
    spl = 1000

    generateLas(js, X_bound=X_bound, Y_bound=Y_bound, splitter=spl)



