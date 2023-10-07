###################################################
## PRoblem 1A
library(AmesHousing)
library(tidyverse)
# data(ames_raw)
ames <- make_ordinal_ames()

# ames <- ames_raw
ames$Remodeled <- ames$Year_Built != ames$Year_Remod_Add

ames$Remodeled <- ames$`Year Built` != ames$`Year Remod/Add`
# ames$Remodeled

set.seed(248)

names(ames)


samped <- sample_n(ames, size = 500)
remod <- filter(samped, Remodeled==TRUE) %>%
  select(Sale_Price)

nonRemod <- filter(samped, Remodeled==FALSE) %>%
  select(Sale_Price)

t.test(remod, nonRemod)
###############################3




###########################
# Problem 2A
set_a <- rnorm(50, 15, 8)

set_b <- rnorm(50, 17, 8)
hist(set_a)
hist(set_b)

t.test(set_a, set_b)
# length(ames_raw)
# 1# ames[[1,3]]


loop <- function(n){
c = numeric(20)
for (x in 1:20){
  n1 <- n
  n2 <- n
  
  mu1 <- 15
  mu2 <- 17
  
  sigma1 <- 8
  sigma2 <- 8
  
  set_a <- rnorm(n1, mu1, sigma1)
  
  set_b <- rnorm(n2, mu2, sigma2)

  res <- t.test(set_a, set_b, "less")
  
  c[x] <- res$p.value < 0.05
}
return(c)
}
  
samp_size <- seq(from = 100, to = 1000, by = 10)

# samp_size <- c(100,200,300,400,500,600,700,800,900)

found <- FALSE

for (idx in 1:length(samp_size)){
  c <- loop(samp_size[idx])
  reject <- sum(c) / length(c)

  
  if(reject >= 0.95){found <- samp_size[idx] 
  
  break}
  
  
  
  
  
  
  
  
}

c = numeric(20)
for (x in 1:20){
n1 <- 1000
n2 <- 1000

mu1 <- 15
mu2 <- 17

sigma1 <- 8
sigma2 <- 8

set_a <- rnorm(n1, mu1, sigma1)

set_b <- rnorm(n2, mu2, sigma2)
# hist(set_a)
# hist(set_b)

res <- t.test(set_a, set_b, "less")

c[x] <- res$p.value < 0.05
}