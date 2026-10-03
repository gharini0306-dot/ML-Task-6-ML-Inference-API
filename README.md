# Task 6: Real-Time ML Inference REST API

## Overview

This project packages a trained classification model as a REST API using FastAPI.

## Model

The API uses the champion model selected in Task 4 for the Breast Cancer Wisconsin classification dataset.

## Endpoints

* `GET /` — API welcome message
* `GET /health` — Health status
* `POST /predict` — Returns a class prediction and probabilities

## Request Format

The `/predict` endpoint accepts 30 numerical features in the same order used during model training.

## Technologies

* Python
* FastAPI
* Scikit-learn
* Joblib
* Pytest
* Docker

## Run Locally

Install dependencies:

`python -m pip install -r requirements.txt`

Start the API:

`python -m uvicorn app.main:app --reload`

Open `http://127.0.0.1:8000/docs` to explore the API.

## Run Tests

`python -m pytest -v`

## Docker

Build the image:

`docker build -t ml-task-6 .`

Run the container:

`docker run -p 8000:8000 ml-task-6`

## Disclaimer

This project is for educational purposes only. Its predictions are not medical advice and must not be used for medical diagnosis.
