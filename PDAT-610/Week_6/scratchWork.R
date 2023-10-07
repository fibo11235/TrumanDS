library(tidyverse)

library(AmesHousing)

ames <- make_ordinal_ames()

boxplot(ames$Sale_Price ~ ames$Overall_Cond, xlab = "Overall Condition", ylab = "Sales Price")

## 1b The Sales price is steadily increasing from very poor to Excellent.  Average seems to have more outliers 
## that are pulling the median and mean sales price higher.  

# 2a
ggplot(ames, aes(Overall_Cond, Sale_Price)) +
 geom_point() +
  # geom_jitter(width = 0.5, height = 0.5) +
  # theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1)) +
 geom_jitter(aes(colour = Year_Built, size = Lot_Area)) +
theme(axis.text.x = element_text(angle = 90)) +
  xlab("Overall Condtion") + 
  ylab("Sales Price")

# 2b
# Year built seems to have a stronger relationship to sales price, as well as lot area.  Larger lots are more 
# expensive




# p + geom_jitter(size = ames$Lot_Area)


