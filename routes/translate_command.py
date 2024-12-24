from fastapi import APIRouter, HTTPException

from models.command import TranslationRequest, TranslationResponse
from src import llm # Assuming this function is defined in llm.py

router = APIRouter()

@router.post("/translate", response_model=TranslationResponse)
async def translate(request: TranslationRequest):
    try:
        command = await llm.translate_to_command(request.text)
        return TranslationResponse(command=command)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))