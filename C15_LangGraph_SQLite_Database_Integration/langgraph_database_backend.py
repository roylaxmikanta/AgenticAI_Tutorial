from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated, Iterator
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.message import add_messages
from dotenv import load_dotenv
import sqlite3
import os

load_dotenv()
hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    huggingfacehub_api_token=hf_token,
    temperature=0.7,
)
llm = ChatHuggingFace(llm=llm)

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

def chat_node(state: ChatState):
    messages = state['messages']
    response = llm.invoke(messages)
    return {"messages": [response]}

conn = sqlite3.connect(database='chatbot.db', check_same_thread=False)
checkpointer = SqliteSaver(conn=conn)

graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)

chatbot = graph.compile(checkpointer=checkpointer)

def invoke_chatbot(messages, thread_id="default"):
    print("invoke_chatbot called with thread_id =", thread_id)
    config = {"configurable": {"thread_id": thread_id}}
    return chatbot.invoke({"messages": messages}, config=config)

def stream_chatbot(messages, thread_id="default") -> Iterator[str]:
    print("stream_chatbot called with thread_id =", thread_id)
    config = {"configurable": {"thread_id": thread_id}}
    for chunk in chatbot.stream(
        {"messages": messages},
        config=config,
        stream_mode="messages"
    ):
        if isinstance(chunk[0], BaseMessage):
            if hasattr(chunk[0], "content"):
                yield chunk[0].content

def retrieve_all_threads():
    all_threads = set()
    for checkpoint_tuple in checkpointer.list(None):
        thread_id = checkpoint_tuple.config.get('configurable', {}).get('thread_id')
        if thread_id:
            all_threads.add(thread_id)
    return list(all_threads)

