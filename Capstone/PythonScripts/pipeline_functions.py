import datetime
import pdal
import json
import numpy as np
from datetime import datetime, timezone
import multiprocessing as mp    

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







if __name__ == "__main__":
    X_bound = [-10099875.0,-10083682.0]
    Y_bound = [4685827.0,4696129.0]
    spl = 1000

    generateLas(js, X_bound=X_bound, Y_bound=Y_bound, splitter=spl)



