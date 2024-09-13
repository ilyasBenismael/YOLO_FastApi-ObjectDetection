# Use the official Python slim image for efficiency
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy requirements.txt into the container
COPY requirements.txt .

# Install the Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project into the container
COPY . .

# Run the FastAPI app using Uvicorn server
CMD ["uvicorn", "app.main:myapp", "--host", "0.0.0.0", "--port", "8000"]
