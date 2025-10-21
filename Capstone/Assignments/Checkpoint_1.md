# Week 1

## Checkpoint

My capstone will be over 3D image segmentation. Specifically, I will apply machine learning algorithms to segment an aerial LiDAR (light detection and ranging) laser scan of a metro area. These LiDAR scans are usually done yearly and are classified at the point level. The approach used will multiscale neighborhoods detailed in this paper. This paper applied a spherical neighborhood method at various radii and downscaling. Aerial scans, however, will usually use a cylindrical neighborhood method for feature extraction. My capstone will use both to compare methods on an aerial LiDAR scan.

I have code written to extract data and perform the necessary feature extraction using radial neighborhoods. Next will be to apply a cylindrical method. This will be done by projecting the points to the horizontal plane and using a 2D radial neighborhood at multiple scales and downsampling.



After feature extraction has been conducted, a random forest classifier will be applied to the point cloud and applied to test data.