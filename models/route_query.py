from pydantic import BaseModel, Field
from typing_extensions import Literal, Optional

class RouteQuery(BaseModel):
    """Route a user query to the most relevant datasource."""

    datasource: Literal["vectorstore", "translate_to_command", "web_search", 'memory', 'agent'] = Field(
        ...,
        description="Given a user chat history choose to route it to web search, robot command translation, or a vectorstore for information related to DTETI like lecturer, advisory board, vision and mission, and contacts. If nothing match please route to agent",
    )
    answer: str = Field(
        description="Binary answer 'yes' or 'no' whether could be answered or not based on provided messages"
    )