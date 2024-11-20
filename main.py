import os
from dotenv import load_dotenv, dotenv_values
from openai import OpenAI
from typing import Annotated
from typing_extensions import TypedDict
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()
apiKey = os.getenv("OPENAI_CHATGPT_APIKEY")

llm = ChatOpenAI(model="gpt-3.5-turbo",
    temperature=0,
    max_tokens=100,
    timeout=None,
    api_key=apiKey)

class State(TypedDict):
    messages: Annotated[list, add_messages]

graph_builder = StateGraph(State)

memory = MemorySaver()

def chatbot(state: State):
    return {"messages": [llm.invoke(state["messages"])]}

graph_builder.add_node("chatbot", chatbot)
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)

graph = graph_builder.compile(checkpointer=memory)
config = {"configurable": {"thread_id": "1"}}

def stream_graph_updates(user_input: str):
    for event in graph.stream({"messages": [("user", user_input)]},config):
        for value in event.values():
            print("Assistant:", value["messages"][-1].content)

while True:
    try:
        user_input = input("User: ")
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break

        stream_graph_updates(user_input)
    except:
        # fallback if input() is not available
        user_input = "Please tell me the problem you are facing.?"
        print("User: " + user_input)
        stream_graph_updates(user_input)
        break