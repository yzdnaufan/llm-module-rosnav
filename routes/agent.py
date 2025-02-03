from fastapi import APIRouter, HTTPException

from models.agent import AgentRequest

from src import graph

router = APIRouter()

@router.post("/agent")
def rag(request: AgentRequest):
    try:
        result = graph.invoke_graph(request.text)
        return result
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except TypeError as te:
        raise HTTPException(status_code=422, detail=str(te))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))