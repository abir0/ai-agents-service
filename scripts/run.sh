#!/bin/bash

# Create logs dir
mkdir -p logs

# Run the API service
nohup python src/run_service.py > logs/service.log 2>&1 &

# Run the Streamlit app
cd src/ui/
nohup streamlit run app.py > ../../logs/app.log 2>&1 &
cd -
