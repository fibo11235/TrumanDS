#!/bin/bash




docker build -t torch -f ./Capstone/build/Dockerfile .
echo "Remove dangling images..."
docker rmi $(docker images -f "dangling=true" -q)
echo "Done."



