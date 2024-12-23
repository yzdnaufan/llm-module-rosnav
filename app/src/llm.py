import os
from typing import Any, Dict

from langchain_community.tools.tavily_search import TavilySearchResults

from langchain_core.messages import HumanMessage, SystemMessage, RemoveMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from langchain_openai import ChatOpenAI

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.prebuilt import create_react_agent

from dotenv import load_dotenv

from app.models.command import Command, TranslationResponse

load_dotenv()

# Define constants
MODEL = 'gpt-3.5-turbo'
TEMPERATURE = 0.5
MAX_RETRIES = 2
MAX_TOKENS = 100

# Ensure required environment variables are set
required_vars = ['LANGCHAIN_API_KEY', 'OPENAI_API_KEY', 'TAVILY_API_KEY' ]
for var in required_vars:
    if var not in os.environ:
        raise EnvironmentError(f"Missing required environment variable: {var}")

# Now you can safely use os.environ to access your environment variables
langchain_api = os.environ['LANGCHAIN_API_KEY']
openai_api = os.environ['OPENAI_API_KEY']

def translate_to_command(text:str) -> Dict[str, Any] : {


}

