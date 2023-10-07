abalone.url <-
  "https://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data"
abalone.data <- read.csv(abalone.url, header=FALSE) #data does not have variable names
names(abalone.data) <- c("sex", "length", "diameter",
                         "height", "weight.whole", "weight.shucked",
                         "weight.viscera", "weight.shell", "rings")
class(abalone.data)

typeof(abalone.data$diameter)

summary(abalone.data[c("diameter","rings")])

mean(abalone.data$diameter[abalone.data$sex=="F"])
# x <- c(1, 2, 3, 4)
# x^2
# w <- c(48, 53, 62, 39)
# m <- cbind(w,x)
# m2 <- matrix(1:8, ncol=2)
# m %*% t(m2)
# 
# t(m)

X <- matrix(abalone.data$rings)
Y <- matrix(abalone.data$diameter)

# help("solve")
x_tx <- solve(t(X) %*% X)
outs <- x_tx %*% t(X) %*% Y

# plot(Y, X)
help("t.test")

# t.test

# et_sex <- subset(abalone.data, sex != "I")

abalone.data$determine_sex = abalone.data$sex != "I"
# undet_sex <- subset(abalone.data, sex == "I")

boxplot(abalone.data$weight.shell ~ abalone.data$det_sex, xlab = "Sex is determined", ylab = "Diameter")




# subset(df, gender == 'M')

abalone.data$det_sex = abalone.data$sex != "I"
ROOT = "/data/Week_2/abalone/abalone.data"
abalone.data <- read.csv(ROOT, header=FALSE) 
names(abalone.data) <- c("sex", "length", "diameter",
                         "height", "weight.whole", "weight.shucked",
                         "weight.viscera", "weight.shell", "rings")

abalone.data$det_sex = abalone.data$sex != "I"

hist(abalone.data$weight.shell ~ abalone.data$det_sex)
boxplot(abalone.data$weight.shell ~ abalone.data$det_sex, xlab = "Sex is determined", ylab = "Diameter")
# sum(abalone.data$det_sex)

# det_sex <- subset(abalone.data, sex != 'I')

# undet_m <- subset(abalone.data, sex == 'I')
install.packages("moments")
library(moments)

boxplot(abalone.data$length ~ abalone.data$det_sex, xlab = "Sex is determined", ylab = "Abalone length")

m_f_t <- subset(abalone.data, det_sex == TRUE)

i_t <- subset(abalone.data, det_sex != TRUE)
?ks.test
ks.test(m_f_t$weight.shell)

ks.test(m_f_t$weight.shell ~ m_f_t$sex, alternative = "greater")

t.test(m_f_t$weight.shell ~ m_f_t$sex, alternative = "greater")

?wilcox.test

wilcox.test(m_f_t$weight.shell, i_t$weight.shell, paired = FALSE, conf.int = TRUE,alternative = "g", conf.level = 0.99)
?t.test



?ks.test
skewness(i_t$weight.shell)
?t.test
# wilcox.test(m_f_t$weight.shell, i_t$weight.shell)

t.test(m_f_t$weight.shell, i_t$weight.shell,alternative = "g", conf.level = 0.99)

t.test(1:10, y = c(7:20))      # P = .00001855
t.test(1:10, y = c(7:20, 200)) # P = .1245    -- NOT significant anymore


