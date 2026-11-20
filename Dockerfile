FROM python:3.12-slim-trixie

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app/src \
    SAUDE_FETAL_ROOT=/app \
    HOME=/tmp

WORKDIR /app

RUN apt-get update \
    && apt-get upgrade -y --no-install-recommends \
    && apt-get install -y --no-install-recommends ca-certificates libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid 10001 model \
    && useradd --uid 10001 --gid model --home-dir /tmp --no-create-home model

COPY docker/requirements.txt /app/docker/requirements.txt
RUN python -m pip install --no-cache-dir -r /app/docker/requirements.txt

COPY configs/experiment.yaml /app/configs/experiment.yaml
COPY src/saude_fetal/ /app/src/saude_fetal/
COPY docker/serve_model.py /app/docker/serve_model.py

USER model
EXPOSE 8080
ENTRYPOINT ["python", "/app/docker/serve_model.py"]
