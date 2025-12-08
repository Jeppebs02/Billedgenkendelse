FROM python:3.11-slim

ARG APP_VERSION=unknown

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    ULTRALYTICS_SETTINGS_DIR=/app/.ultralytics \
	APP_VERSION=$APP_VERSION

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

# Build context is Python-code, so paths start there
WORKDIR /app

COPY Billedegenkendelse/requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && pip install -r /app/requirements.txt

# Copy the whole Python-code tree into /app
COPY . /app

# IMPORTANT: run from the Billedegenkendelse folder so relative 'models/...' works
WORKDIR /app/Billedegenkendelse

# (Optional) ensure models are present in the image:
# If your repo contains the files below under Billedegenkendelse/models, you're done.
# If not, mount them as a volume (see docker-compose below).

EXPOSE 5001
CMD ["python", "application.py"]
