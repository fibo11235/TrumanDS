# Truman Data Science 

This is a general purpose repository for the [Truman State University data science masters program](https://www.truman.edu/majors-programs/graduate-studies/data-science-2/).  The folders are named based on the coursework I found most interesting or helpful to post to github.  

## Capstone

## Introduction

The goal of this project is to segment two different types of point clouds using two different methods for comparison.  The first method will use a pointwise radial method, where neighboring points will be within a radius $r$ of a point, $p_0$.   Since point clouds are usually very large datasets, the neighbors will be downsampled as the neighborhood expands to ensure computational stability.  Pointwise  

### Neighborhoods

There are different definitions of neighborhoods besides radial neighborhoods.  The two other popular methods include **k nearest neighbors** and **cylindrical neighbors**. This study focuses on radial and cylindrical neighborhoods on two different types of point clouds.  Cylindrical neighborhoods are mostly used for aerial scans, while radial neighborhoods and k nearest neighborhoods are primarily used on mobile or terrestrial data, this study will test both neighborhood types on both mobile and aerial scans.  [Thomas et al]((https://ieeexplore.ieee.org/document/8490990) demonstrated that geometrically uniform neighborhoods like radial neighbors give better results than a *k nearest neighbors* approach, while *k nearest neighbors* are less computationally expensive.  


$$
\begin{equation}
N(p_0) = \left\{p \mid \Vert p - p_0 \Vert \leq r \right\}
\end{equation}
$$

for some radius, $r$.  


