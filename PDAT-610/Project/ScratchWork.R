library(AmesHousing)
library(dplyr)
library(tidyverse)
library(knitr) #For generating tables in the results section
library(xtable) #For generating tables in the results section
library(patchwork) # For creating the plots using ggplot


### Load data
data("ames_raw")

#### Compute Age of house
amesfixed <- ames_raw %>% 
  mutate(Age = 2011 - `Year Built`)
#### Compute remodelling age
amesfixed <- amesfixed %>% 
  mutate(RemodelAge = 2011 - `Year Remod/Add`)

### Some housekeeping code to fix field names
amesfixed <- rename(amesfixed, Gr.Liv.Area=`Gr Liv Area`)
amesfixed <- rename(amesfixed, Garage.Area=`Garage Area`)

### Filter based on the outliers specified in the DeCock paper
amesUse <- amesfixed %>% 
  filter(Gr.Liv.Area <= 4000)

### Keep the ames data we want and drop NAs
amesUse <- amesUse %>%
  select(Gr.Liv.Area, Garage.Area, Age, RemodelAge, SalePrice) %>%
  drop_na()




### Create the model and create table in results section
model <- lm(SalePrice ~ Gr.Liv.Area + Garage.Area + Age + RemodelAge, amesUse)

model %>%
  summary() %>%
  xtable() %>%
  kable()

library(knitr)
library(xtable)

# lm(SalePrice ~ Gr.Liv.Area + Garage.Area + RemodelAge + Age , amesUse) %>%
model %>%
  summary() %>%
  xtable() %>%
  kable()

xx <- summary(model)


library(patchwork)


amesUse <- amesfixed %>%
  select(MS.Zoning,Gr.Liv.Area, Garage.Area,Lot.Area, Age, RemodelAge, SalePrice) %>%
  drop_na()
amesUse <- amesUse %>%
  filter(Gr.Liv.Area <= 4000) 
amesUse %>%
  group_by(MS.Zoning) %>% 
  summarize(avearage = mean(Lot.Area, na.rm = TRUE))

boxplot(amesUse$Gr.Liv.Area, xlab = "Lot Area", main = "Boxplot of Lot Area")
boxplot(amesUse$Age)

plot(amesUse$Gr.Liv.Area, amesUse$SalePrice)

plot(amesUse$Lot.Area, amesUse$SalePrice)
plot(amesUse$Age, amesUse$SalePrice)

plot(amesUse$RemodelAge, amesUse$SalePrice)

plot(amesUse$Garage.Area, amesUse$SalePrice)









ggplot(amesUse, aes(x=Garage.Area, y=SalePrice)) + 
  xlab("Garage Area") + 
  ylab("Sales Price") +
  geom_point() + 
  geom_smooth(method=lm)

library(knitr)
library(xtable)

lm(SalePrice ~ Gr.Liv.Area+Lot.Area + Garage.Area + RemodelAge + Age , amesUse) %>%
  summary() %>%
  xtable() %>%
  kable()

model <- lm(SalePrice ~ Gr.Liv.Area + Garage.Area + RemodelAge + Age , amesUse) 
summary(model)

xxx<-summary(model)

xxx$r.squared
xxx$sigma



summa <- summary(model)

### Get R^2
summa$r.squared

### Get standard deviation of residuals

summa$sigma

# Or:
dof<-5
sqrt(sum((summa$residuals)^2) / (length(summa$residuals) - dof))


xxx$fstatistic[3]
dof<-5
summary(model)$r.squared
lm(SalePrice ~ Gr.Liv.Area + Garage.Area + RemodelAge + Age , amesUse) %>%
  summary() %>%
  xtable() %>%
  kable()

summary(lm(SalePrice ~ Gr.Liv.Area + Garage.Area + RemodelAge + Age , amesUse) )
  

summary(lm(SalePrice ~ Garage.Area, amesUse))

model <- lm(SalePrice ~ Gr.Liv.Area+Lot.Area + Garage.Area + RemodelAge + Age, data = amesUse)

model <- lm(SalePrice ~ Gr.Liv.Area, amesUse)
summary(model)

# 
# KimDataPhys <- KimData %>% 
#   select(Semester, Shoe.Size, Height, Weight, Handed, Gender)




ames.fields


avg.rom <- ames_raw %>% 
  group_by(Street) %>% 
  summarize(Average.Quality = mean(`Overall Qual`, na.rm = TRUE), STD.Quality = sd(`Overall Qual`, na.rm= TRUE))

colnames(ames_raw)




plot(ames_raw$`Gr Liv Area`, ames_raw$SalePrice,xlab = "Ground Living Area", ylab = "Sales Price",main = "Sales price vs ground living area")
boxplot(ames_raw$`Gr Liv Area`)

ames_raw_fixed <- ames_raw %>%
  filter(`Gr Liv Area` < 4000)

boxplot(ames_raw_fixed$`Gr Liv Area`)

plot(ames_raw_fixed$`Gr Liv Area`, ames_raw_fixed$SalePrice,xlab = "Ground Living Area", ylab = "Sales Price",main = "Sales price vs ground living area")
  
### a bit more of a skeleton
### 