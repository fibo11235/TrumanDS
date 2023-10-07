###### Answer to question 3

# Summary of Titanic dataset
#install.packages("carData") already installed
library(carData)
summary(TitanicSurvival)

#### Answer to question 4
# Table command to show number of survivors in each class
### First determine if there are any NAs in each of the pertinent columns
sum(is.na(TitanicSurvival$survived))
sum(is.na(TitanicSurvival$passengerClass))
#### No NAs in these columns
surv_table <- table(TitanicSurvival$passengerClass, TitanicSurvival$survived)
surv_table


#### Text portion  for answer ####
# The number of people in 3rd class have many more people who did not survive (25% survival rate)
# compared to first and second (62% and 43%, respectively)

#### Answer to question 5
#### Make a histogram of ages
hist(TitanicSurvival$age, xlab = "Age of titanic passengers",main = "Histogram of Passenger Ages")

#### Text portion  for answer ####

########### The histogram appears to be skewed right




#### Answer to question 6
boxplot(TitanicSurvival$age ~ TitanicSurvival$survived, xlab = "Survived", ylab = "Age", main= "Age Distribution of passengers by survival")


#### Text portion  for answer ####
# There seems to be very little difference in the age distribution.  The Age distribution of those who didn't survive has more outlier values.

