# Overview
Objective of this project is to create a regression model based on ANN for generating concrete mix. Model can be trained either inside Docker container or locally via venv because CPU is enough. 

# Train via Docker
1. Build your docker image with 
```bash
docker build -t inzynierka .
```

2. Run your container from image
```bash
docker run inzynierka
```
3. To get inside Docker container
type:
```bash
docker ps
```
in terminal to get info about all running containers. After that type:
```bash
docker exec -it <CONTAINER_ID> /bin/sh
```
# Train via venv
Create virtual envirnoment
```bash
python3 -m venv .venv
source .venv/bin/activate
```
Remember to upgrade pip to the current version. Install requirements.
```bash
pip install --upgrade pip
pip install -r requirements.txt
```
Call training
```bash
python trainer.py
```


# Documentation of docker
```
https://docs.docker.com/engine/reference/commandline/build/
```