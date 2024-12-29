FROM tensorflow/tensorflow:2.18.0-gpu-jupyter

WORKDIR /app

COPY ./requirements.txt /app/requirements.txt

RUN pip install --upgrade pip
# RUN pip install --upgrade -r /app/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt

COPY ./* /app/

CMD ["tail", "/dev/null"]