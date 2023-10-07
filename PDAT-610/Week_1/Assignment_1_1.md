PDAT 610G M1 Assign1- Introduction to R


Module 1 Assignment 1-1
------------------------------------------------------------------------
 1. Finish walking through this lab. (The actual lab starts on page 2 (right after question #7.)


 2. Use the + button to create a new R script.  Save it with a filename that
    looks like "lab0_yournamehere.R".  Put all the commands you use to
    answer the next questions into the new script file.  Answers to questions
    that aren't code can be set off by #, just as done in this file.


 3. Use the "summary" command to get a basic summary of each of the variables
    in the TitanicSurvival data set.  Note that the type of summary you get
    depends on the data type of each variable. You need to install Titanic package first as below. 
install.packages("carData")
library(carData)
summary(TitanicSurvival)
	

 4. Use the ? operator to read about the "table" command.  Then create a
    table that shows the number of people who survived and didn't survive in
    each passenger class.  What pattern do you see in the data?


 5. Use the "hist" command to make a histogram of ages.  Do you notice
    anything interesting in the shape of the histogram?


 6. Use the ? operator to learn about the "boxplot" command.  Hint: Often the
    examples at the end of the help file are especially helpful.  Make
    side-by-side boxplots that show the distribution of ages for survivors
    and for those who didn't survive.  Does it appear that there is a
    difference in the age distribution of the two groups?


 7. Save your script file with only the commands you used and the answers to the questions that were asked (question 3,4,5, and 6).  Copy-and-paste that RScript onto the word file and submit it as your assignment 1 on Blackboard. Submit the output for question 3,4,5, and 6 as well.

















#The actual lab starts here


#To create a new script document, click the icon with the green + sign on the right corner in R studio and select a new "R #Script."
#Copy-and-paste all of the text (here to the end of the lab) into a blank RScript Tab and you are ready to start.
#(Click Control-A, then Control-C, then go to the blank RScript and click Control-V


# Lab 0 Introduction to RStudio and the R programming language
# This is an R script. Lines that start with as hashtag # are comments 
# and not executed # by R. 
# You _could_ run this script in its entirety, but I'd like you to
# walk through it step-by-step, following the directions.


# We will load this first (0th?) lab using the not-very-snazzy method of 
# copy-and-paste. Soon, we will move to a different kind of script called 
# RMarkdown. RMarkdown is nice because it turns your output directly 
# into .pdf, .htm, or .doc files. 
# Lab 1 will get to that soon enough.


# R software
#------------------------------------------------------------------------
# You want to download your own copy of R and RStudio to your own
# computer by going to http://rstudio.com 
# Make sure you download the standard packages that come with it.


# RStudio v R software
#------------------------------------------------------------------------
# In general the software language we are using is called R.
# RStudio is a graphical user interface (GUI) that makes R easier to use.
# RStudio has 4 "corners" that each serve a different purpose.
# The ones on the left are for code, and the ones on the right keep track of things.


# R Console 
#------------------------------------------------------------------------
# The "Console" is the window where you can interactively execute R commands
# and see their output. If you were to use R without RStudio, the console
# would be the only window you'd see.
#
# Click inside the console window if you haven't already. You should notice
# your blinking cursor next to the blue > symbol.  This means that R is waiting
# for a command to execute. Type 2+2 and then hit the <ENTER> key.  You should
# see the result 4, next to the symbol [1], like this:
# > 2+2
# [1] 4
# Now try 6/2, 6*2, 6^2 and 6/0. Mathematically, we might say that 6/0 is
# undefined, but R represents it as Inf, for infinity.


# Script Window
#------------------------------------------------------------------------
# You should be reading this lab in the Script Window. To create a new script
# document, click the icon with the green + sign and select a new "R Script."
# An R script is  a "scratchpad" where you can enter commands at your leisure
# and then run them in the Console window when you are ready. As soon as you
# make a new Script document, you should save it into your project with a name,
# so that it automatically saves things for you. Anything you do in the script
# window can be saved, and most everything in the console window is not.
#
# To run a command on a single line, hold <CTRL> and hit <ENTER>. Try it on
# the non-commented line below:


2 + 7


# You should see 2 + 7 copied into the console window, as well as the result
# of the calculation.
#
# To run multiple lines from your script, highlight them all with the cursor,
# then hit <CTRL><ENTER>. Try it:


2 + 3
2 * 3
2 ^ 3


# You should get 5, 6 and 8 as output.
#
# It is nice to do your coding in a Script Window so that you can save it for
# later, copy and paste, and create reproducible analyses.




# The 2nd column
#------------------------------------------------------------------------
# The right column has two more "corners" that we'll use more soon.
# The top-right "Environment" tab remembers things and makes it easy 
# to import data from Excel or other files. The "History" tab tracks what
# you've done today, which can be handy.
# The bottom-left has five tabs that will show plots, help screens, as well
# as remembering what folder you are saving things in.




# Data Sets
#------------------------------------------------------------------------
# When we started RStudio, a whole bunch of standard data sets were loaded
# into our current memory.  To find out which data sets are there, type the
# command data() or hit <CTRL><ENTER> on the line below.


data()


# Another window in the upper left will appear, (called “R data sets”) listing
# all data sets currently loaded.  We are interested in the dataset
# USJudgeRatings, so type ?USJudgeRatings to find help on what this data set
# is all about.  Notice that the help information on this data set shows up in
# the lower right panel of Studio.  R is case sensitive, so be sure to pay
# attention to capital letters when needed in the data title.  


?USJudgeRatings


# Data Frames
#------------------------------------------------------------------------
# The most common way for R to store a data set is in a "data frame." Data
# frames are always formatted so that each column represents one variable and
# each row represents one individual or "csae."  Data frames usually contain
# "raw" data, not pre-summarized tables.
#
# To see the contents of a data frame, you can type its name:


USJudgeRatings


# That might be a lot of output! If you just want to see the column names and
# organization of the data frame, you can use the "head" command. (Try it.)


head(USJudgeRatings)


#View is another option, notice the capital V
View(USJudgeRatings)


# You can see that the data frame has several variables (columns). If you want
# to refer to only one of them, you can use the $ operator. Suppose we wanted
# to look only at the overall retention ratings for each judge. We could type:


USJudgeRatings$RTEN


# We will be doing much of our data cleaning and processing using a package
# called tidyverse (more on that soon). However, the built-in system, using 
# square brackets can be very powerful, too.
# What does this show you?
USJudgeRatings$RTEN[USJudgeRatings$RTEN>8]


# We will be doing much of our data cleaning and processing using a package
# called tidyverse (more on that soon). However, the built-in system, using 
# square brackets can be very powerful, too.




# Tab Completion (The newer version of Rstudio does this automatically)
#------------------------------------------------------------------------
# You might start to feel that you're often typing very long names into R.  To
# make life easier, R uses "tab completion."  Start typing the name of a
# variable or object, then hit the <TAB> key.  RStudio will show you the
# possible completions of that word, and you can use the arrow keys to choose
# from that list of options.
#
# Try typing US in the console, then hitting <TAB> to find possible
# completions.
#
# Try typing USJ in the console, then hitting <TAB>.  If only one name matches
# what you've typed, you won't have to select from a menu.


# Descriptive Statistics
#------------------------------------------------------------------------
# You can use R to calculate descriptive statistics, like means, standard
# deviations, quartiles, etc. for any vector of numbers.  Here are three
# useful commands to calculate statistics for the judge's overall retention
# ratings.  See what each of them do.


mean(USJudgeRatings$RTEN)
sd(USJudgeRatings$RTEN)
summary(USJudgeRatings$RTEN)


# If you're ever in doubt about what a command does, you can use the ? help
# feature. For example,


?sd


# Plots
#------------------------------------------------------------------------
# Now let's construct a histogram of judge's retention ratings.  You'll notice
# that graphs appear in the bottom right window in RStudio.


hist(USJudgeRatings$RTEN)


# Looking at the graph, you might say that ratings are right-skewed.  Most
# judges have a rating between 7 and 9, with relatively few judges trailing
# down into lower ratings.
#
# It's possible to change features of plots. In this case, let's change the
# title and x axis labels with the "main" and "xlab" options.  Many commands
# have optional arguments that are specified in the same way: name=option.


hist(USJudgeRatings$RTEN, main="Histogram of Retention Ratings", xlab="Rating")


# There are many types of graphs you could make.  Let's make a scatter plot to
# look for a relationship between two variables.  Do you think that the rating
# of a judge's physical ability would be related to the ranking of a judge's
# integrity?  They wouldn't necessarily be related, would they?...


plot(USJudgeRatings$PHYS, USJudgeRatings$INTG)


# These were examples of R's basic graphics commands. Base R is cool in that
# it tries to guess the right kind of chart to make, and it often does a pretty
# good job, making a boxplot, scatterplot, or dotplot as it thinks you want it.
# However, knowing what's going on inside 


# There are other graphics systems as well. In a later lab, we'll learn 
# another graphics system (ggplot) that's even more flexible and snazzy.


# Packages
#------------------------------------------------------------------------
# R's basic functionality can be extended by the use of "packages" that
# define new commands, new analytical techniques, and new data sets.  There
# are two steps to using packages in R:
#
# (1) install: Download from the internet and install on your computer.
# (2) load library: Load the new code into R's working memory for use.
#
# Both steps have their own commands. You need only install a package once,
# but you must load it as a library every time you run R.
#
# The "car" package is a nice package that's not about cars (the name stands
# for Companion to Applied Regression).  It contains lots of nice data sets,
# including the TitanicSurvival data set we'll use in the questions at the end.
# Here's how you would install and load it:


install.packages("carData")                      # You need to do this only once.
library(carData)                                 # Load the package into memory.




# On the bottom right portion of the RStudio window, there's a "Packages" tab
# that makes installation, loading, and unloading of packages easier.  You
# should be able to scroll down the list until you find the car package.  See
# what happens when you uncheck and check the package, but make sure it ends
# up checked (loaded).


# We'll use the tidyverse package. It's actually a whole
# bunch of specific packages for particular purposes, but you can load them all
# by just typing
install.packages("tidyverse")  
library(tidyverse)


# If you use R long enough, you may have to update them, but not during this semester.


# Variable Assignment
#------------------------------------------------------------------------
# You can assign numeric values, vectors (discussed below), or even more
# complicated data structures to a named variable in R. The left arrow <- is
# the assignment operator in R.  Variable names must start with a letter and
# can contain letters (upper and lower case), numbers and a few other symbols
# like _ (underscore) and . (period).  Once you've defined a variable, you can
# use it just as you would the original value. See below:


a <- 2
a^2
a.long.label.for.the.number.5 <- 5
a.long.label.for.the.number.5 - a


# If you don't like typing <- all the time, you can also use = as an
# assignment operator (frowned upon in R circles) or use the shortcut <ALT>-
# (hold ALT key and hit minus key) to insert an arrow.  (Fun fact: In ancient
# times, terminals had an actual "arrow" key.)


# Vectors
#------------------------------------------------------------------------
# A "vector" in R is on ordered list of values (all of the same type---see
# "types" below). Each column of a data frame is a vector of values.  You can
# define vectors yourself using the c (combine) command or the : operator.
# Run the commands below to see what each does.


a <- c(1, 5, 3, 9, 10)
a
b <- 1:5
b


# Many functions in R are "vectorized," which means that they will operate on
# each element of a vector and return a vector as a result.  This will come in
# handy in making formulas and computations look much nicer than they
# otherwise would.  More on that later, but at least two examples for now.


a+b
b^2


# Note that the elements of a and b are added component-wise and return a
# vector of the same length. b^2 squares each element of b separately.


# Data Types
#------------------------------------------------------------------------
# There are many data types in R, but we'll look at only three of them here:
# Numeric, Character and Factor.  To get a visual aid, we'll use one of the
# data sets that was loaded with the "car" package: data on individual
# passengers on the Titanic and whether they survived.


#install.packages("carData")
#library(carData)
head(TitanicSurvival)


# Of all these variables, only "age" is numeric.  It consists of a set of
# numeric values for age in years.  Name is a character variable.  Each entry
# in the name list can be any collection of characters, and most entries are
# different than the others. 
#
# The variables "survived", "sex" and "passengerClass" are "factor" variables.
# While they look like character variables, they have a special structure.
# There are only a few possible values these variables can take on, and those
# values define groups of passengers.  For example 1st class passengers are
# one group.  R can use factor variables in many of its data analysis
# functions.















Fall 2022 Update
