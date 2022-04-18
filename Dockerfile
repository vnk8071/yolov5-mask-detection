FROM python:3.8-slim

COPY ./requirements.txt /app/requirements.txt

WORKDIR /app

RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6 libgl1 -y
RUN pip install -r requirements.txt

COPY /config/ /app/config/
COPY /models/ /app/models/
COPY /static/ /app/static/
COPY /templates/ /app/templates/
COPY /utils/ /app/utils/
COPY ./app.py /app/app.py

EXPOSE 5000
ENTRYPOINT [ "python" ]
CMD ["app.py" ]