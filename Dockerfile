FROM ubuntu:latest

WORKDIR /inzynierka

RUN apt update
RUN apt install python3 -y
RUN apt-get -y install python3-pip

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY dataset.py .
COPY trainer.py .
COPY model.py .
COPY ./datasets ./datasets

CMD ["python3", "./trainer.py"]