import uuid
from fastapi import FastAPI, Request, File, UploadFile, Form, BackgroundTasks
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import pandas as pd
from contextlib import asynccontextmanager

from agent import get_exchange_rate, init_ollama, run_agent

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Initializing Ollama...")
    init_ollama()
    print("Ollama initialized successfully")
    yield
    # shutdown
    print("Shutting down")


# create the FastAPI app
app = FastAPI(lifespan=lifespan)

# mount templates directory
templates = Jinja2Templates(directory="templates")


# home route
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("upload.html", {"request": request})


# upload route
@app.post("/upload", response_class=HTMLResponse)
async def upload_file(request: Request, file: UploadFile = File(...)):
    try:
        if not file.filename.endswith(".csv"):
            return templates.TemplateResponse(
                "upload.html",
                {
                    "request": request,
                    "message": "Please upload a csv file",
                    "message_type": "dange",
                },
            )
        random_filename = str(uuid.uuid4())
        file_path = UPLOAD_DIR / random_filename
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)

        return templates.TemplateResponse(
            "upload.html",
            {
                "request": request,
                "message": "File uploaded successfully!",
                "message_type": "success",
                "filename": random_filename,
            },
        )
    except Exception as e:
        return templates.TemplateResponse(
            "upload.html",
            {
                "request": request,
                "message": f"An error occurred:{str(e)}",
                "message_type": "danger",
            },
        )


# process file endpoint
@app.post("/process", response_class=HTMLResponse)
async def process_file(
    request: Request, background_tasks: BackgroundTasks, filename=Form(...)
):
    file_path = UPLOAD_DIR / filename
    if not file_path.exists():
        return templates.TemplateResponse(
            "upload.html",
            {
                "request": request,
                "message": "File not found. Please upload again",
                "message_type": "dangeer",
            },
        )

    # start background task
    LOG_FILE = UPLOAD_DIR / f"{filename}.log"
    background_tasks.add_task(run_agent, LOG_FILE, file_path)

    return templates.TemplateResponse(
        "upload.html",
        {
            "request": request,
            "message": f"started processing. Log will update below",
            "message_type": "info",
            "filename": filename,
            "show_log": True,
        },
    )


# Log route
@app.get("/log/{filename}")
async def get_log(filename: str):
    LOG_FILE = UPLOAD_DIR / f"{filename}.log"
    if LOG_FILE.exists():
        with open(LOG_FILE, "r") as f:
            log_content = f.read()

        return PlainTextResponse(log_content)

    return PlainTextResponse("No log yet")
