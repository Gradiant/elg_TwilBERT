FROM python:3.6


COPY TWilBert /app/TWilBert
COPY requirements.txt pythonpath.sh config /app/

RUN pip3 install --upgrade pip
RUN pip3 install -r /app/requirements.txt

WORKDIR /app/

RUN bash /app/pythonpath.sh

ENV LANG="C.UTF-8" \
    LC_ALL="C.UTF-8"

EXPOSE 8866

WORKDIR /app/TWilBert/

CMD ["python3", "serve.py"]
RUN ["python3", "-c", "from init_model import Initializer; Initializer()"]

ENV TRANSFORMERS_OFFLINE=1
