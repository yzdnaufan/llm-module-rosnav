from typing import Dict, List
from pydantic import BaseModel

class Command(BaseModel):
    """
    Command model representing a parsed command from text.

    Attributes:
        text (str): The original text of the command.
        action (str): The action to be performed.
        confidence (float): The confidence level of the command parsing.
        parameters (Dict[str, str]): Additional parameters for the command.
    """
    text: str
    action: str
    confidence: float
    parameters: Dict[str, str]

class TranslationResponse(BaseModel):
    """
    TranslationResponse model representing the response of the translation.

    Attributes:
        commands (List[Command]): A list of parsed commands.
    """
    commands: List[Command]

class TranslationRequest(BaseModel):
    """
    TranslationRequest model to represent a request for text translation.
    Attributes:
        text (str): The text to be translated.
    """
    text: str