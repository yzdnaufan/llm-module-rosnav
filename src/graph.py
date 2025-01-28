import os

from typing import Any, Dict
from typing_extensions import List, TypedDict, Literal

from langchain import hub

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma

from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import create_retriever_tool

from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings

from langchain_text_splitters import RecursiveCharacterTextSplitter

from langgraph.graph import START, StateGraph, END

from pydantic import BaseModel, Field

from dotenv import load_dotenv

from models.command import Translation

import yaml

load_dotenv()

# Define constants
AGENT_MODEL = os.environ.get('AGENT_MODEL')
GENERATE_MODEL = os.environ.get('GENERATE_MODEL')
GRADE_MODEL = os.environ.get('GRADE_MODEL')
EMBEDDING_MODEL = os.environ.get('EMBEDDING_MODEL')
TEMPERATURE = os.environ.get('TEMPERATURE')
MAX_RETRIES = os.environ.get('MAX_RETRIES')
MAX_TOKENS = os.environ.get('MAX_TOKENS')

# Prompt
# Load the YAML file
with open('prompt.yaml', 'r') as file:
    prompt = yaml.safe_load(file)


# Ensure required environment variables are set
required_vars = ['LANGCHAIN_API_KEY', 'OPENAI_API_KEY', 'TAVILY_API_KEY' ]
for var in required_vars:
    if var not in os.environ:
        raise EnvironmentError(f"Missing required environment variable: {var}")

# Now you can safely use os.environ to access your environment variables
langchain_api = os.environ['LANGCHAIN_API_KEY']
openai_api = os.environ['OPENAI_API_KEY']
tavily_api = os.environ['TAVILY_API_KEY']

# Initialize the ChatOpenAI object

llm = ChatOpenAI(
    api_key=openai_api,
    model=MODEL,
    temperature=TEMPERATURE,
    max_retries=MAX_RETRIES,
    max_tokens=MAX_TOKENS
)

embeddings = OpenAIEmbeddings(
    api_key=openai_api,
    model=EMBEDDING_MODEL
)

grade_llm = ChatOpenAI(
    api_key=openai_api,
    model=GRADE_MODEL,
    temperature=TEMPERATURE,
    max_retries=MAX_RETRIES,
    max_tokens=MAX_TOKENS
)

# # Initialize the TavilySearchResults object
web_search_tool = TavilySearchResults(api_key=tavily_api, max_results=3)

##############
####Router####
##############
class RouteQuery(BaseModel):
    """Route a user query to the most relevant datasource."""

    datasource: Literal["vectorstore", "translate_to_command", "web_search", 'agent'] = Field(
        ...,
        description="Given a user question choose to route it to web search, robot command translation, or a vectorstore. If no match route to agent",
    )
structured_llm_router = llm.with_structured_output(RouteQuery)
route_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", prompt['routing-prompt']),
        ("human", "{question}"),
    ]
)
question_router = route_prompt | structured_llm_router

#############
##Retriever##
#############
docs_path = ('docs')
docs_list = []
for filename in os.scandir(docs_path):
    loader = PyPDFLoader(filename.path)
    for page in loader.lazy_load():
        docs_list.append(page)
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=100, chunk_overlap=50
)
doc_splits = text_splitter.split_documents(docs_list)
# Add to vectorDB
vectorstore = Chroma.from_documents(
    documents=doc_splits,
    collection_name="rag-chroma",
    embedding=OpenAIEmbeddings(),
)
retriever = vectorstore.as_retriever()
retriever_tool = create_retriever_tool(
    retriever,
    "retrieve_about_dteti",
    "Search and return information about DTETI, lecturers, advisory boards, facility, vision and mission, and it's workers",
)
tools = [retriever_tool]

####################
##Retrieval Grader##
####################
class GradeDocuments(BaseModel):
    """Binary Score relevance check for documents retrieved"""

    binary_score: str = Field(description="Documents are relevant to the query,'yes' or 'no'")
structured_grader = llm.with_structured_output(GradeDocuments)
grade_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", prompt['grader-prompt']),
        ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
    ]
)
retrieval_grader = grade_prompt | structured_grader

###############
###RAG Chain###
###############
rag_chain = hub.pull('rlm/rag-prompt') | llm | StrOutputParser()

####################
###Query Rewriter###
####################
rewrite_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", prompt['query-rewriter-prompt']),
        ("human", "Initial question: {question} \n\n Improved question: "),
    ]
)
rewrite_chain = rewrite_prompt | llm | StrOutputParser()

#################
###Graph State###
#################
class GraphState(TypedDict):
    """
    GraphState is a TypedDict that represents the state of a graph.
    Attributes:
        question (str): A question provided by the user.
        generated (str): A string indicating the generated LLM string.
        document (List[str]): A list of strings representing the document retrieved.
        retries (int): An integer representing the number of retries.
    """
    question: str
    generated: str
    documents: List[str]
    retries: int = 0

################
###Graph Flow###
################
def agent(state):
    """
    Invokes the agent model to generate a response based on the current state. Given
    the question, it will decide to retrieve using the retriever tool, or simply end.

    Args:
        state (messages): The current state

    Returns:
        dict: The updated state with the agent response appended to messages
    """
    print("---CALL AGENT---")
    messages = state["question"]
    model = ChatOpenAI(temperature=0, streaming=True, model="gpt-4-turbo") | StrOutputParser()
    response = model.invoke(messages)
    # We return a list, because this will get added to the existing list
    return {"generated": [response]}

def retrieve(state):
    """
    Retrieve relevant documents from the vectorstore.

    Args:
        state (GraphState): The current state

    Returns:
        dict: The updated state with the retrieved documents
    """
    print("---RETRIEVE---")
    question = state["question"]
    retry = state["retries"]
    print("RETRY:", retry)
    docs = retriever.invoke(question)
    return {"documents": [doc.page_content for doc in docs]}

def generate(state):
    """
    Generate an answer based on the retrieved documents.

    Args:
        state (GraphState): The current state

    Returns:
        dict: The updated state with the generated answer
    """
    print("---GENERATE---")
    question = state["question"]
    docs = state["documents"]
    # RAG chain
    result = rag_chain.invoke({'context': docs, 'question': question})
    return {"question": question , "document": docs, "generated": result}

def grade_docs(state):
    """
    Grade the relevance of the retrieved documents.

    Args:
        state (GraphState): The current state

    Returns:
        dict: The updated state with the graded relevance
    """
    print("---GRADE---")
    question = state["question"]
    docs = state["documents"]

    filtered_docs = []
    # Score each docs
    for d in docs:
        doc_txt = d
        score = retrieval_grader.invoke({"question": question, "document": doc_txt})
        if score.binary_score == "yes":
            print('---DOC RELEVANT---')
            filtered_docs.append(d)
        else:
            print('---DOC NOT RELEVANT---')
            continue
    return {"documents": filtered_docs, "question": question}

def rewrite_query(state):
    """
    Rewrite the user query to be more optimized for the vectorstore search.

    Args:
        state (GraphState): The current state

    Returns:
        dict: The updated state with the rewritten query
    """
    print("---REWRITE---")
    question = state["question"]
    
    result = rewrite_chain.invoke({"question": question})
    return {"question": result, "retries": state["retries"] + 1}

def translate_to_command(state):
    """
    Translate the user query to a robot command.

    Args:
        state (GraphState): The current state

    Returns:
        dict: The updated state with the translated command
    """
    print("---TRANSLATE---")
    question = state["question"]
    result = llm.with_structured_output(Translation).invoke(question)
    return {"generated": result, "question": question}

def web_search(state):
    """
    Perform a web search for the user query.

    Args:
        state (GraphState): The current state

    Returns:
        dict: The updated state with the web search results
    """
    print("---WEB SEARCH---")
    question = state["question"]
    try :
        docs = web_search_tool.invoke({"query": question})
        web_res = '\n\n'.join([d['content'] for d in docs])
        web_res = Document(page_content=web_res)
    except Exception as e:
        print(e)
        web_res = "Error: Web search failed please try in a moment."
        web_res = Document(page_content=web_res)

    return {"question": question, "documents": web_res}

######## Edges ########

def route_question(state):
    """
    Route the user question to the most relevant datasource.

    Args:
        state (GraphState): The current state

    Returns:
        str: The datasource to route the question to
    """
    print("---ROUTE---")
    print("--STATE--", state)
    question = state["question"]
    result = question_router.invoke({"question": question})
    if result.datasource == 'vectorstore':
        print('---VECTORSTORE---')
        return 'vectorstore'
    elif result.datasource == 'translate_to_command':
        print('---TRANSLATE---')
        return 'translate_to_command'
    elif result.datasource == 'web_search':
        print('---WEB SEARCH---')
        return 'web_search'
    else:
        return 'agent'
    
def decide_to_generate(state):
    """
    Decide whether to generate an answer based on the relevance of the documents.

    Args:
        state (GraphState): The current state

    Returns:
        str: The decision to generate or not
    """
    print("---DECIDING TO GENERATE---")
    docs = state["documents"]
    if state["retries"] >= 2:
        print('---RETRIES EXCEEDED WEB SEARCH WILL BE USED---')
        return 'web_search'
    if len(docs) > 0:
        print('---RESULT: GENERATE---')
        return 'generate'
    else:
        print('---RESULT: REWRITE---')
        return 'rewrite_query'

def chat(text:str) -> Dict[str, Any] : 
    """
    Chat with the robot using the LLM model.

    Args:
        text (str): The text to chat with the robot.

    Returns:
        Dict[str, Any]: The response from the robot.
    """
    structured_llm = llm.with_structured_output(HumanMessage)
    response = structured_llm.invoke(text)
    return {
        'text' : text,
        'response' : response
    }
workflow = StateGraph(GraphState)

# Define the nodes
workflow.add_node("web_search", web_search)  # web search
workflow.add_node("retrieve", retrieve)  # retrieve
workflow.add_node("translate_to_command", translate_to_command)  # translate to command
workflow.add_node("grade_docs", grade_docs)  # grade documents
workflow.add_node("generate", generate)  # generate
workflow.add_node("rewrite_query", rewrite_query)  # rewrite query
workflow.add_node("agent", agent)  # agent

# Build graph
workflow.add_conditional_edges(
    START,
    route_question,
    {
        "web_search": "web_search",
        "vectorstore": "retrieve",
        "translate_to_command": "translate_to_command",
        'agent': 'agent'
    },
)
workflow.add_edge("web_search", "generate")
workflow.add_edge("retrieve", "grade_docs")
workflow.add_conditional_edges(
    "grade_docs",
    decide_to_generate,
    {
        "rewrite_query": "rewrite_query",
        "generate": "generate",
        "web_search": "web_search",
    },
)
workflow.add_edge("rewrite_query", "retrieve")
workflow.add_edge("generate", END)
workflow.add_edge("translate_to_command", END)
workflow.add_edge("agent", END)

# Compile
app = workflow.compile()

def invoke_graph(message:str) -> Dict[str,Any] :
    """
    Invokes the graph processing application with the given message.
    Args:
        message (str): The message to be processed by the graph application.
    Returns:
        Dict[str, Any]: The result from the graph application.
    """
    
    state = {
        "question": message,
        "retries": 0
    }
    result = app.invoke(state)
    return result