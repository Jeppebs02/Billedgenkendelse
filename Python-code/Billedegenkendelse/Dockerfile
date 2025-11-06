# 1) Use glibc-based image so mediapipe & torch wheels work
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# 2) System deps needed for opencv-python (headless) etc.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

# 3) Workdir is the project root for this service
WORKDIR /app

# 4) Copy only requirements first (better layer caching)
#    NOTE: path is from the *build context* (Python-code)
COPY Billedegenkendelse/requirements.txt /app/requirements.txt

# 5) Install deps. Pin mediapipe to a version that supports py3.11 wheels.
RUN pip install --upgrade pip \
 && pip install -r /app/requirements.txt

# 6) Copy the rest of the code (models, logic, app, etc.)
COPY . /app

# 7) Expose Flask port and run the API
EXPOSE 5001
CMD ["python", "Billedegenkendelse/application.py"]
