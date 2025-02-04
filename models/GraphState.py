from datetime import datetime, timedelta
from typing import List
from langgraph.graph import MessagesState

#################
###Graph State###
#################

class GraphState(MessagesState):
    """
    GraphState is a MessageState extended class that represents the state of a graph.
    Attributes:
        messages (MessageState): A list of messages for memory.
        question (str): A question provided by the user.
        generated (str): A string indicating the generated LLM string.
        document (List[str]): A list of strings representing the document retrieved.
        retries (int): An integer representing the number of retries.
        timestamp : A time to indicate session 
        expiry : An expiry time length
    """
    question: str
    generated: str
    documents: List[str]
    retries: int = 0
    timestamp: datetime
    expiry: timedelta