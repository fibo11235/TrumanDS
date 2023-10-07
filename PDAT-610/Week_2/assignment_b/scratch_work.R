library(tidyverse)
ROOT = "/data/Week_2/assignment_b/Clean-KimData.csv"
Clean.KimData <- read.csv(ROOT)
KimData <- Clean.KimData


KimDataYear <- KimData %>%
  mutate(Year=round(Semester / 2))
# 
# mutate(Model = ifelse(Stimulus == "426789", "child",
#                       ifelse(Stimulus == "426790", "child",
#                              ifelse(Stimulus == "426783", "adult",
#                                     ifelse(Stimulus == "426784", "adult", "no"))))) %>%
#   
#   mutate(Emotion = ifelse(Stimulus == "426789", "angry",
#                           ifelse(Stimulus == "426790", "happy",
#                                  ifelse(Stimulus == "426783", "happy",
#                                         ifelse(Stimulus == "426784", "angry", "no")))))
?if_else
?case_when
# KimDataYear <- KimDataYear %>%
#   mutate(ClassStatus=if_else(Year >= 3), "senior",
#          if_else((Year>=2 & Year <3), "junior",
#                  if_else((year))

KimDataYear <- KimDataYear %>% 
 mutate(Class.Name = case_when(Year < 1 ~ "Freshman",
            Year < 2 ~ "Sophomore",
            Year < 3 ~ "Junior",
            Year >= 3 ~ "Senior"))

KimDataBMI <- KimDataYear %>%
  mutate(BMI = 703*(Weight / Height^2)) %>%
  mutate(Obese = BMI > 30)
                 
                 
KimDataBMI %>% group_by(Class.Name) %>%
  summarize(n = n(), Obese.Percent = 100*mean(Obese==TRUE, na.rm=TRUE))              



# KimData.NotNa <- KimData %>% 
#   filter(!is.na(Weight)) %>% 
#   filter(!is.na(Height))

KimDataBMI <- KimData %>% 
  mutate(Year=round(Semester/2)) %>%
  mutate(BMI = 703*(Weight / Height^2)) %>%
  mutate(Obese = BMI > 30)
  
percent<- KimDataBMI %>% 
  filter(!is.na(BMI)) %>% 
  group_by(Year) %>% 
  summarize(Percent.Obese = sum(Obese==TRUE) / sum(!is.na(BMI)))


KimDataBMI[c("Semester","Year")]



KimDataPhys <- KimData %>% 
  select(Semester, Shoe.Size, Height, Weight, Handed, Gender)

head(KimDataPhys)




KimDataNoHand <- KimDataPhys %>% select(-Handed)

KimNumData <- select(KimData, c(1,3,5:7, 12, 13:22))
View(KimNumData)

KimDataP4<- mutate(KimDataPhys, Gender=as.factor(sub("other", NA, Gender)))

# 
# head(KimDataBMI)
# ?count
# nrow(KimDataBMI)
