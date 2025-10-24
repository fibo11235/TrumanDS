#!/bin/bash


start=`date +%s`

end=`date +%s`


docker build -t r_image -f ./DockerfileR/Dockerfile .
echo "Remove dangling images..."
docker rmi $(docker images -f "dangling=true" -q)
echo "Done"
echo "Total Time: " 
echo $((end-start))



