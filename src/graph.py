import os

from typing import Any, Dict

from dotenv import load_dotenv

from datetime import datetime, timezone, timedelta


from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma

from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain.agents.agent_toolkits import create_conversational_retrieval_agent

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, RemoveMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import create_retriever_tool

from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings

from langchain_text_splitters import RecursiveCharacterTextSplitter

from langgraph.graph import START, StateGraph, END

from langgraph.checkpoint.memory import MemorySaver

from pydantic import BaseModel, Field

import yaml

from models.command import Translation
from models.route_query import RouteQuery
from models.GraphState import GraphState


load_dotenv(dotenv_path='.env')

# Ensure required environment variables are set
required_vars = ['LANGCHAIN_API_KEY', 'OPENAI_API_KEY', 'TAVILY_API_KEY' ]
for var in required_vars:
    if var not in os.environ:
        raise EnvironmentError(f"Missing required environment variable: {var}")

# Define the configuration
config = {"configurable": {"thread_id": "abc123"}}

# Define constants
AGENT_MODEL = os.environ.get('AGENT_MODEL') or 'gpt-3.5-turbo'
GENERATE_MODEL = os.environ.get('GENERATE_MODEL')  or 'gpt-3.5-turbo' # Agent that used in all flow of the graph
GRADE_MODEL = os.environ.get('GRADE_MODEL') or 'gpt-3.5-turbo'
EMBEDDING_MODEL = os.environ.get('EMBEDDING_MODEL') or 'text-embedding-3-large'
TEMPERATURE = os.environ.get('TEMPERATURE') or 0.5
MAX_RETRIES = os.environ.get('MAX_RETRIES') or 2
MAX_TOKENS = os.environ.get('MAX_TOKENS') or 100

# Prompt
# Load the YAML file
with open('prompt.yaml', 'r') as file:
    prompt = yaml.safe_load(file)


# Now you can safely use os.environ to access your environment variables
langchain_api = os.environ.get('LANGCHAIN_API_KEY')
openai_api = os.environ.get('OPENAI_API_KEY')
tavily_api = os.environ.get('TAVILY_API_KEY')

# Initialize the ChatOpenAI object

llm = ChatOpenAI(
    api_key=openai_api,
    model=GENERATE_MODEL,
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

memory = MemorySaver()

# # Initialize the TavilySearchResults object
web_search_tool = TavilySearchResults(api_key=tavily_api, max_results=3, api_wrapper=TavilySearchAPIWrapper())

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
    chunk_size=1000, chunk_overlap=100
)
doc_splits = text_splitter.split_documents(docs_list)
# Add to vectorDB
vectorstore = Chroma.from_documents(
    documents=doc_splits,
    collection_name="rag-chroma",
    embedding=OpenAIEmbeddings(model=EMBEDDING_MODEL)
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
structured_grader = grade_llm.with_structured_output(GradeDocuments)
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
rag_chain = ChatPromptTemplate([("system", prompt['prompt-rag-chain'])]) | llm 

####################
###Query Rewriter###
####################
rewrite_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", prompt['query-rewriter-prompt']),
        ("human", "Initial question: {question} \n\n Improved question: "),
    ]
)
rewrite_chain = rewrite_prompt | llm 

################
###Graph Flow###
################

######### Nodes #########


# First Call Model
# Define the function that calls the model
def call_model(state):
    print("---CALL MODEL----")

    # system_prompt = (
    #     "You are a helpful assistant. "
    #     "Answer all questions to the best of your ability. "
    #     "The provided chat history includes a summary of the earlier conversation."
    # )
    # system_message = SystemMessage(content=system_prompt)
    message_history = state["messages"][:-1]  # exclude the most recent user input

    # Summarize the messages if the chat history reaches a certain size
    if len(message_history) >= 8:
        last_human_message = state["messages"][-1]
        summary_prompt = (
            "Distill the above chat messages into a summary message. "
            "Include as many specific details as you can."
        )
        summary_message = llm.invoke(
            message_history + [HumanMessage(content=summary_prompt)]
        )

        # Re-add user message
        human_message = HumanMessage(content=last_human_message.content)
        # Delete messages that we no longer want to show up
        delete_messages = [RemoveMessage(id=m.id) for m in state["messages"]]
        # Call the model with summary & response
        # response = llm.invoke([system_message, summary_message, human_message])
        # message_updates = [summary_message, human_message, response] + delete_messages

        message_updates = [summary_message, human_message] + delete_messages
        print(message_updates)
        return {"messages": message_updates}
    else:
        # message_updates = llm.invoke([system_message] + state["messages"])
        return {"messages": state["messages"]}

    # print(message_updates)
    # return {"messages": message_updates}


# Agent

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
    messages = state["messages"]
    model = ChatOpenAI(temperature=0.3, model="gpt-4o")
    response = model.invoke(messages)
    # We return a list, because this will get added to the existing list
    return {"generated": [response.content], "messages" : response}

# def delete_messages(state):
#     messages = state["messages"]
#     tnow = datetime.now(timezone.utc)
#     d = state['expiry']
#     if len(messages) > 10:
#         return {"messages": [RemoveMessage(id=m.id) for m in messages[:-4]]}
#     if state['timestamp'] - tnow > d :
#         return {"messages": [RemoveMessage(id=m.id) for m in messages], "timestamp": datetime.now(timezone.utc)}

def retrieve(state):
    """
    Retrieve relevant documents from the vectorstore.

    Args:
        state (GraphState): The current state

    Returns:
        dict: The updated state with the retrieved documents
    """
    print("---RETRIEVE---")
    question = state["messages"][-1]
    retry = state["retries"]
    print("RETRY:", retry)
    docs = retriever.invoke(question.content)
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
    question = state["messages"][-1]
    docs = state["documents"]
    # RAG chain
    result = rag_chain.invoke({'context': docs, 'question': question.content})
    return {"question": question.content , "document": docs, "generated": result.content, 'messages': result}

def grade_docs(state):
    """
    Grade the relevance of the retrieved documents.

    Args:
        state (GraphState): The current state

    Returns:
        dict: The updated state with the graded relevance
    """
    print("---GRADE---")
    question = state["messages"][-1]
    docs = state["documents"]

    filtered_docs = []
    # Score each docs
    for d in docs:
        doc_txt = d
        score = retrieval_grader.invoke({"question": question.content, "document": doc_txt})
        if score.binary_score == "yes":
            print('---DOC RELEVANT---')
            filtered_docs.append(d)
        else:
            print('---DOC NOT RELEVANT---')
            continue
    return {"documents": filtered_docs}

def rewrite_query(state):
    """
    Rewrite the user query to be more optimized for the vectorstore search.

    Args:
        state (GraphState): The current state

    Returns:
        dict: The updated state with the rewritten query
    """
    print("---REWRITE---")
    question = state["messages"][-1]
    
    result = rewrite_chain.invoke({"question": question.content})
    return {"question": result.content, "retries": state["retries"] + 1, "messages": result}

def translate_to_command(state):
    """
    Translate the user query to a robot command.

    Args:
        state (GraphState): The current state

    Returns:
        dict: The updated state with the translated command
    """
    print("---TRANSLATE---")
    question = state["messages"]
    result = llm.with_structured_output(Translation).invoke(question)
    return {"generated": result, 'messages': AIMessage(content="Oke, sudah saya terima perintahnya")}

def web_search(state):
    """
    Perform a web search for the user query.

    Args:
        state (GraphState): The current state

    Returns:
        dict: The updated state with the web search results
    """
    print("---WEB SEARCH---")
    question = state["messages"][-1]
    try :
        docs = web_search_tool.invoke({"query": question.content})
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
    llm = ChatOpenAI(model='gpt-4o', temperature=0.3)
    structured_llm_router = llm.with_structured_output(RouteQuery)

    # Prompt
    system = """You tasked to route a user query to a memory or vectorstore or web search or robot command translation tools  .
    If you could answer it immediately please answer it. Priorities answering based on memory and in the language used.
    The vectorstore contains documents related to DTETI lecturer, vision and mission, and its advisory boards.
    Use the vectorstore for questions on these topics. Otherwise, use web-search.
    Use robot command translation tool when provided query is to navigate.
"""

    question = state["messages"]
    if question[0] is not SystemMessage:
        question.insert(0, SystemMessage(content=system))
    
    result = structured_llm_router.invoke(question)

    print("Is Answerable by memory:", result.answer)

    if result.answer == 'yes' :
        print('---MEMORY---')
        return 'memory'
    
    if result.datasource == 'vectorstore':
        print('---VECTORSTORE---')
        return 'vectorstore'
    elif result.datasource == 'translate_to_command':
        print('---TRANSLATE---')
        return 'translate_to_command'
    elif result.datasource == 'web_search':
        print('---WEB SEARCH---')
        return 'web_search'
    elif result.datasource == 'agent':
        print('---AGENT---')
        return 'agent'
    else :
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
    if state["retries"] >= 1:
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
workflow.add_node(call_model) # first init
workflow.add_node("web_search", web_search)  # web search
workflow.add_node("retrieve", retrieve)  # retrieve
workflow.add_node("translate_to_command", translate_to_command)  # translate to command
workflow.add_node("grade_docs", grade_docs)  # grade documents
workflow.add_node("generate", generate)  # generate
workflow.add_node("rewrite_query", rewrite_query)  # rewrite query
workflow.add_node("agent", agent)  # agent
# workflow.add_node(delete_messages) # delete messages memory

# Build graph

workflow.add_edge(START, "call_model")

workflow.add_conditional_edges(
    "call_model",
    route_question,
    {
        "web_search": "web_search",
        "vectorstore": "retrieve",
        "translate_to_command": "translate_to_command",
        'memory': 'agent',
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
# workflow.add_edge("generate", "delete_messages")
# workflow.add_edge("translate_to_command", "delete_messages")
# workflow.add_edge("agent", "delete_messages")
workflow.add_edge("generate", END)
workflow.add_edge("translate_to_command", END)
workflow.add_edge("agent", END)

# workflow.add_edge("delete_messages", END)


# Compile
app = workflow.compile(checkpointer=memory)

# Compile
app = workflow.compile(checkpointer=memory)

def invoke_graph(message:str) -> Dict[str,Any] :
    """
    Invokes the graph processing application with the given message.
    Args:
        message (str): The message to be processed by the graph application.
    Returns:
        Dict[str, Any]: The result from the graph application.
    """
    try :
        app.get_state(config=config).values['timestamp']
        state = {
            "question": message,
            "messages": [HumanMessage(content=message)], 
            "retries": 0
        }
    except:
        time = datetime.now(timezone.utc)
        delta = timedelta(minutes=5)
        state = {
            "question": message,
            "messages": [HumanMessage(content=message)], 
            "retries": 0,
            "timestamp" : time,
            "expiry": delta
        }

    result = app.invoke(state, config=config)
    return result