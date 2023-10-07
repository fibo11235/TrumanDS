
library(AmesHousing)

AmesData <- make_ames()


data("ames_geo")

data("ames_raw")
data("ames_school_districts_sf")
data("ames_schools_geo")
data("ames_schools_geo")



# df.combined <- merge(ames_raw, ames_geo, "PID")
# 
# df.combined$Latitude
# 
# 
# 
# AmesData$w3ifelse <- if_else(AmesData$Year_Built >= 2000, 1, 0)
# head(AmesData$w3ifelse, 10)
# 
# 
# for (i in 1:nrow(AmesData)) {
#   
#   if (is.na(AmesData$Year_Built[i])){
#     AmesData$w3for[i] <- NA
#   } else if(AmesData$Year_Built[i] >= 2000){
#     AmesData$w3for[i] <- 1
#   } else {
#     AmesData$w3for[i] <- 0 
#   }
# }
# 
# head(AmesData$w3for, 10)
# 
# past2000 <- function(x) {
#   if (is.na(x)) return (x)
#   else if ( x >=2000) return (1)
#   else return (0)
# }
# 
# AmesData$w3apply <- sapply(AmesData$Year_Built, past2000)
# 
# library(sf)
# library(terra)
# library(tmap)
# 
# 
# 
# tm_shape()
# 
# tm_shape(p.sf) + tm_fill(col="SalePrice", style="quantile",n=8,palette = "Greens") + tm_legend(outside=TRUE)
# ames_raw$SalePrice
# 
# df.combined
# 
# 
# 
# df.combined
# 
# p.sf <- st_as_sf(df.combined, coords = c("Longitude", "Latitude"), crs = 4326) 
# 
# s.sp <- as_Spatial(p.sf)
# 
# 
# p.sf$geometry
# 
# plot(p.sf["SalePrice"],graticule = TRUE, axes= TRUE)
# 
# trans <- st_transform(p.sf, crs=6463)
# 
# grid <- expand.grid()
# 
# library(gstat)
# library(sp)
# 
# SpatialPointsDataFrame(p.sf)
# 
# trans$geometry
# 
# library(spdep)
# 
# 
# dist <- as.matrix(dist(cbind(trans$geometry)))
# 
# plot(st_geometry(trans$geometry))
# 
# plot(st_geometry(p.sf$geometry))
# p.sf
# 
# p <- vect(system.file("ex/lux.shp", package="terra"))
# 
# plot(p)
# p.sf
# install.packages("isdas")

  

