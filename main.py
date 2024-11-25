from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
from dotenv import load_dotenv
from openai import OpenAI
from typing import Annotated
from typing_extensions import TypedDict
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from fastapi.middleware.cors import CORSMiddleware

# Load environment variables
load_dotenv()
apiKey = os.getenv("OPENAI_CHATGPT_APIKEY")

# Initialize FastAPI app
app = FastAPI(title="Chatbot API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize ChatOpenAI
llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0,
    max_tokens=100,
    timeout=None,
    api_key=apiKey
)

# Define State class
class State(TypedDict):
    messages: Annotated[list, add_messages]

# Initialize graph builder and memory
graph_builder = StateGraph(State)
memory = MemorySaver()

# Define chatbot function
def chatbot(state: State):
    return {"messages": [llm.invoke(state["messages"])]}

# Build graph
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)
graph = graph_builder.compile(checkpointer=memory)

# Define request model
class ChatRequest(BaseModel):
    message: str
    thread_id: str = "1"

# Define response model
class ChatResponse(BaseModel):
    response: str

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(chat_request: ChatRequest):
    try:
        config = {"configurable": {"thread_id": chat_request.thread_id}}
        
        # Process the message through the graph
        response = None
        for event in graph.stream(
            {"messages": [("user", chat_request.message)]},
            config
        ):
            for value in event.values():
                response = value["messages"][-1].content
        
        if response is None:
            raise HTTPException(status_code=500, detail="Failed to generate response")
        
        return ChatResponse(response=response)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

# Run with: uvicorn main:app --reload
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 
    # app.run(debug = True)
    
# import os
# from dotenv import load_dotenv, dotenv_values
# from openai import OpenAI
# from typing import Annotated
# from typing_extensions import TypedDict
# from langchain_openai import ChatOpenAI
# from langchain.prompts import ChatPromptTemplate
# from langgraph.graph import StateGraph, START, END
# from langgraph.graph.message import add_messages
# from langgraph.checkpoint.memory import MemorySaver

# load_dotenv()
# apiKey = os.getenv("OPENAI_CHATGPT_APIKEY")

# llm = ChatOpenAI(model="gpt-3.5-turbo",
#     temperature=0,
#     max_tokens=100,
#     timeout=None,
#     api_key=apiKey)

# class State(TypedDict):
#     messages: Annotated[list, add_messages]

# graph_builder = StateGraph(State)

# memory = MemorySaver()

# def chatbot(state: State):
#     return {"messages": [llm.invoke(state["messages"])]}

# graph_builder.add_node("chatbot", chatbot)
# graph_builder.add_edge(START, "chatbot")
# graph_builder.add_edge("chatbot", END)

# graph = graph_builder.compile(checkpointer=memory)
# config = {"configurable": {"thread_id": "1"}}

# def stream_graph_updates(user_input: str):
#     for event in graph.stream({"messages": [("user", user_input)]},config):
#         for value in event.values():
#             print("Assistant:", value["messages"][-1].content)

# while True:
#     try:
#         user_input = input("User: ")
#         if user_input.lower() in ["quit", "exit", "q"]:
#             print("Goodbye!")
#             break

#         stream_graph_updates(user_input)
#     except:
#         # fallback if input() is not available
#         user_input = "Please tell me the problem you are facing.?"
#         print("User: " + user_input)
#         stream_graph_updates(user_input)
#         break