FROM python:3.6

COPY ./requirements.txt /src/requirements.txt

RUN pip install -r /src/requirements.txt

COPY ./ /src
