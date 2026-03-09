# 1. Use an official lightweight Python image
FROM python:3.10-slim

# 2. Set the working directory inside the container
WORKDIR /app

# 3. Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copy the project folders into the container
COPY src/ ./src/
COPY models/ ./models/
COPY Data/ ./Data/  

# 5. Expose the port FastAPI runs on
EXPOSE 8080

# 6. Command to run the API using Uvicorn
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8080"]