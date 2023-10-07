### Load data
library(AmesHousing)
library(tidyverse)
library(ggplot2)

ames <- make_ordinal_ames()


### Get age from 2011
amesfixed <- ames %>%
  mutate(Age = 2011 - Year_Built)
### Get age of remodelling from 2011
amesfixed <- amesfixed %>%
  mutate(RemodelAge = 2011 - Year_Remod_Add)
### Housekeeping functions
# amesfixed <- rename(amesfixed, Lot.Area = `Lot Area`)
# amesfixed <- rename(amesfixed, Gr.Liv.Area=`Gr Liv Area`)
# amesfixed <- rename(amesfixed, Garage.Area=`Garage Area`)
# amesfixed <- rename(amesfixed, MS.Zoning=`MS Zoning`)
### Remove outliers as specificed in DeCook
amesUse <- amesfixed %>%
  filter(Gr_Liv_Area <= 4000 & Year_Built >= 1950)

amesUse <- amesUse %>%
  select(Gr_Liv_Area, Garage_Area, Age, RemodelAge, Sale_Price, House_Style, Roof_Style) %>%
  drop_na()




