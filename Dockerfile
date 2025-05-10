# Dockerfile
FROM python:3.10-slim

# 📦 Install OS packages needed for building wheels
USER root
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential gcc libssl-dev && \
    rm -rf /var/lib/apt/lists/*

# 📦 Set working directory
WORKDIR /app

# 📦 Copy requirements and install Python packages (including Ray)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 📦 Install Antithesis Python SDK (with cffi)
RUN pip install --no-cache-dir antithesis cffi --break-system-packages

# 📦 Copy project code
COPY . .

# 📦 Copy docker-compose.yaml into the correct runtime path
COPY opt/config/docker-compose.yaml /opt/config/docker-compose.yaml

# 📦 Copy your entrypoint Python file
COPY entrypoints/entrypoint.py /entrypoints/entrypoint.py

# 📦 Expose Ray ports (you can expand this later if needed)
EXPOSE 6379 8265

# ⚡ Define the entrypoint to be the Antithesis setup checker
ENTRYPOINT ["python3", "/entrypoints/entrypoint.py"]

# (Optional: eventually split head / worker images separately)
