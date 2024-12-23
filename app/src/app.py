import json
from fastapi import FastAPI
from fastapi import Request
from fastapi.templating import Jinja2Templates
from fastapi import APIRouter
from routes.translate_command import router as translate_router

app = FastAPI()
templates = Jinja2Templates(directory="templates")

app.include_router(translate_router)

@app.get("/")
def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})