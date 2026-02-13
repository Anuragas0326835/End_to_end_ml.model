# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the model artifacts into the container at /app
COPY gym_rf_model.pkl .
COPY scaler.pkl .
COPY encoders.pkl .

# Copy the application code folder
COPY End_to_end_ml.model/ End_to_end_ml.model/

# Set the working directory to where the Flask app is located so relative paths (../) work
WORKDIR /app/End_to_end_ml.model

# Make port 9000 available to the world outside this container
EXPOSE 9000

# Run flask_app.py when the container launches
CMD ["python", "flask_app.py"]