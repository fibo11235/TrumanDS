#!/bin/bash


start=`date +%s`

end=`date +%s`


docker build -t test -f ./PDAT-610/build/Dockerfile .
echo "Remove dangling images..."
docker rmi $(docker images -f "dangling=true" -q)
echo "Done"
echo "Total Time: " 
echo $((end-start))



