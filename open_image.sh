#!/bin/bash
#################
echo "Open a window at http://0.0.0.0:8787/auth-sign-in?appUri=/"


#docker run --rm -ti -v $(pwd)/PDAT-610:/data/ -p 8787:8787 rocker/rstudio
docker run --rm -ti -v $(pwd)/PDAT-610:/data/ -p 8787:8787 test


