# 1. Mandated Base Image (Non-Negotiable)
FROM ubuntu:22.04

# 2. Prevent interactive prompts and set Python optimizations
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# 3. Install Python, Pip, and Build Essentials
# Necessary because raw Ubuntu 22.04 does not include the Python stack
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 4. Set the working directory
WORKDIR /app

# 5. Copy requirements and install
# Leveraging layer caching for faster builds
COPY requirements.txt .
RUN pip3 install --no-cache-dir --upgrade pip && \
    pip3 install --no-cache-dir -r requirements.txt

# 6. Copy the application
COPY . .

# 7. Expose the mandated port
EXPOSE 8000

# 8. Define the run command (Standard binding to 0.0.0.0:8000)
CMD ["python3", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]