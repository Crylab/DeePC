# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the pyproject.toml and poetry.lock files to the working directory
COPY pyproject.toml poetry.lock ./

# Install Poetry
RUN pip install poetry

# Install dependencies
RUN poetry install --no-root

# Copy the rest of the application code to the working directory
COPY source ./source
COPY main.py .

# Create the img directory
RUN mkdir img

# Specify the command to run on container start
CMD ["poetry", "run", "python", "main.py"]