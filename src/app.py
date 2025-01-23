import json
from fastapi import FastAPI
from fastapi import Request
from fastapi.templating import Jinja2Templates

from routes import agent 

app = FastAPI()
templates = Jinja2Templates(directory="templates")

app.include_router(agent.router)

@app.get("/")
def read_root(request: Request):
    return json.dumps({
        "Hello": "World",
        "request": "You requested at: " + str(request.url)
    })