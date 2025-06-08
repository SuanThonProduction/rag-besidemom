FROM python:3.11-slim

# Set the working directory
WORKDIR /app

# Copy the requirements file to the container
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code to the container
COPY . /app

# Expose the port FastAPI will run on
EXPOSE 8000

# Command to run the application (from rag_api.py)
CMD ["uvicorn", "rag_api:app", "--host", "0.0.0.0", "--port", "8000"]
