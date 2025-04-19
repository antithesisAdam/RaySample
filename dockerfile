# Dockerfile
FROM docker.io/rayproject/ray:2.44.1-python3.10

USER root
WORKDIR /app

# 1) Install your Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 2) Copy in your code
COPY . .

# 3) Expose Ray’s ports
EXPOSE 6379 8265

# 4) Default command just prints help; we’ll override in Docker Compose
CMD ["ray", "start", "--help"]
