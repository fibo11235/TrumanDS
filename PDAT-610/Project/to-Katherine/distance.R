library(AmesHousing)
library(geodist) # install.packages("geodist")
## Load ames
ames <- make_ordinal_ames()
## Load school data
data("ames_schools_geo")
### First calculate the distance to the closest school (in Kilometers)
dist <- geodist_vec(x1=ames_schools_geo$Longitude, y1= ames_schools_geo$Latitude,
                    x2=ames$Longitude, y2=ames$Latitude,
                    measure = "geodesic") / 1000
### get minimum distance
mdist <- apply(dist, 2, min)
ames$Distance <- mdist

ames$Distance
