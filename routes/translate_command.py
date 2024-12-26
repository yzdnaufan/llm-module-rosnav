from fastapi import APIRouter, HTTPException

from models.command import TranslationRequest
from src import llm # Assuming this function is defined in llm.py

router = APIRouter()

@router.post("/translate")
def translate(request: TranslationRequest):
    try:
        command = llm.translate_to_command(request.text)
        return command
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))