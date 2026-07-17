# -------------------------
# Base Image
# -------------------------
FROM python:3.11-slim

# -------------------------
# Environment
# -------------------------
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# -------------------------
# System Dependencies
# -------------------------
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    git \
    && rm -rf /var/lib/apt/lists/*

# -------------------------
# Working Directory
# -------------------------
WORKDIR /app

# -------------------------
# Install Python Dependencies
# -------------------------
COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip

# Install CPU-only PyTorch first
RUN pip install --no-cache-dir \
    torch \
    --index-url https://download.pytorch.org/whl/cpu

# Install the rest of the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# -------------------------
# Copy Project
# -------------------------
COPY . .

# -------------------------
# Create Persistent Folders
# -------------------------
RUN mkdir -p uploads
RUN mkdir -p data/chroma_db
RUN mkdir -p data/raw
RUN mkdir -p data/processed

# -------------------------
# Expose Port
# -------------------------
EXPOSE 8000

# -------------------------
# Start FastAPI
# -------------------------
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]