from typing import Dict, List, Optional, Literal

from typing_extensions import Annotated, TypedDict
from pydantic import BaseModel, Field

class CommandParameters(TypedDict):
    """
    CommandParameters model to represent the parameters of a command.
    """
    target: Annotated[Optional[Literal['lecturer_room', 'lab_ai', 'coworking_space']], None, "The target of place of the command, using variable naming style."]
    context: Annotated[Optional[str], None, "The context of the command."]
    # intent: Annotated[Optional[str], ..., "The intent explanation of the command."]

class Command(TypedDict):
    """
    Command model to represent a command.
    """
    isNav: Annotated[bool, ..., "Is the command a navigation command?"]
    parameters: Annotated[CommandParameters, ..., "The parameters of the command."]

class Translation(TypedDict):
    """
    Translation model to represent a translation.
    Attributes:
        text (str): The translated text.
        source_language (str): The source language of the text.
        target_language (str): The target language of the text.
    """
    source_language: Annotated[str, ...,"The source language of the text."]
    command: Annotated[Command, ..., "The command to be executed."]
    confidence: Annotated[float, ..., "The confidence value 0 - 1.0 of the translation into command for robot."]

class AgentRequest(BaseModel):
    """
    RAGRequest model to represent a request for rag-based qna.
    Attributes:
        text (str): The question to be answered.
    """
    text: str

