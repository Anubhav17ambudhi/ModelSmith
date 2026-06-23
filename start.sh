#!/bin/bash

# Start the FastAPI backend internally
uvicorn app.main:app --host 127.0.0.1 --port 8000 &

# Start the Streamlit frontend publicly
streamlit run frontend/app.py --server.port $PORT --server.address 0.0.0.0
