FROM python:3.12.6-slim-bullseye

RUN apt-get update && apt-get install -y git ssh

COPY requirements.txt .
RUN pip install -r requirements.txt

ADD data /home/data
ADD models /home/models
ADD utils /home/utils
ADD configs /home/configs
COPY main.py /home/
COPY main_secnet.py /home/

WORKDIR /home/