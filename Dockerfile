# Dockerfile
FROM docker.io/python:3.10-slim

# (optional) install OS packages needed to build any wheels
USER root
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      build-essential gcc libssl-dev && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy your requirements (must include ray) and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your code
COPY . .

# Copy entrypoint logic
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Expose Ray ports
EXPOSE 6379 8265

ENTRYPOINT ["/entrypoint.sh"]
