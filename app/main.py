"""
Author:     mikechngwk
Project:    Text predictor
Date:       2024-May-01
"""
from fastapi import FastAPI
from app.routers import predict

app = FastAPI()

app.include_router(predict.router)


@app.get("/")
def root():
    return {"message": "AI-powered next sentence prediction is up and running!"}
