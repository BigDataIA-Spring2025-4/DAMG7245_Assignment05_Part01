# Use Python 3.9 slim image as base with specific platform
FROM --platform=linux/amd64 python:3.12.8-slim

# Set working directory
WORKDIR /app

# Install manually all the missing libraries
# RUN apt-get update
# RUN apt-get install -y gconf-service libasound2 libatk1.0-0 libcairo2 libcups2 libfontconfig1 libgdk-pixbuf2.0-0 libgtk-3-0 libnspr4 libpango-1.0-0 libxss1 fonts-liberation libappindicator1 libnss3 lsb-release xdg-utils

# Install Chrome
# RUN wget https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb
# RUN dpkg -i google-chrome-stable_current_amd64.deb; apt-get -fy install

# Copy requirements first for layer caching
COPY requirements.txt .
COPY .env /app/.env

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY ./backend /app/backend
# COPY ./frontend /app/frontend
COPY ./features /app/features
COPY ./services /app/services

# Expose port
EXPOSE 8080

# Run FastAPI application
# Command to run the application
CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8080"]