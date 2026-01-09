FROM python:3.11-slim

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the core package and app code
COPY sg_solver ./sg_solver
COPY app ./app
COPY board_detection ./board_detection

# Install the sg_solver package properly
COPY pyproject.toml .
RUN pip install -e .

# Default port for Fly.io
EXPOSE 8080

CMD ["uvicorn", "app.server:app", "--host", "0.0.0.0", "--port", "8080"]
