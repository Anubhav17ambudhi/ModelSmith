#!/bin/bash

# Start the Celery worker in the background
celery -A Model_Training.worker celery_app worker --loglevel=info &

# Start the FastAPI backend
uvicorn app.main:app --host 0.0.0.0 --port $PORT
