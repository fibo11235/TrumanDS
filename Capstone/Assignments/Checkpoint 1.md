# Week 1

## Checkpoint

My capstone will be over 3D image segmentation. Specifically, I will apply machine learning algorithms to segment an aerial LiDAR (light detection and ranging) laser scan of a metro area. These LiDAR scans are usually done yearly and are classified at the point level. The approach used will multiscale neighborhoods detailed in this [paper](https://ieeexplore.ieee.org/document/8490990). Thomas et al. applied a spherical neighborhood method at various radii and downscaling to conduct 3D feature extraction. Aerial scans, however, will usually use a cylindrical neighborhood method for feature extraction. My capstone will use both to compare methods on an aerial LiDAR scan.  Feature extraction will be done on the covariance matrix of the points found in the neighborhood.  The eigenvalues and eigenvectors from the covariance matrix will be used to calculate geometric properties such as planarity, curvature, etc.  

I have code written to extract data and perform the necessary feature extraction using radial neighborhoods. Next will be to apply a cylindrical method. This will be done by projecting the points to the horizontal plane and using a 2D radial neighborhood at multiple scales and downsampling.



After feature extraction has been conducted, a random forest classifier will be applied to the point cloud and applied to test data.