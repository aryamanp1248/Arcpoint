from dotenv import load_dotenv
load_dotenv()

import os
import subprocess

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

subprocess.run(["python", os.path.join(BASE_DIR, "generate_requests.py")])

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes import router

app = FastAPI(title="ModelPilot Context API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)

@app.get("/")
def root():
    return HTMLResponse(content=open(os.path.join(BASE_DIR, "index.html"), encoding="utf-8").read())