# Dockerfile
FROM python:3.12-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy code and models
COPY src/ /app/src/
COPY models/ /app/models/

EXPOSE 8000

# THIS LINE FIXES THE ERROR: use python -m uvicorn instead of bare "uvicorn"
CMD ["python", "-m", "uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]