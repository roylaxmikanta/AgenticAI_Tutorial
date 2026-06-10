from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated, Iterator
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.messages import BaseMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.message import add_messages
from dotenv import load_dotenv
import os
load_dotenv()
hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    huggingfacehub_api_token=hf_token,
    temperature=0.7
)
llm = ChatHuggingFace(llm=llm)


class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

def chat_node(state: ChatState):
    messages = state['messages']
    response = llm.invoke(messages)
    return {"messages": [response]}

# Checkpointer
checkpointer = InMemorySaver()

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
    """Yield streaming text deltas from the underlying LLM.

    Notes:
    - This function yields *strings only* (may be empty for some chunks).
    - `thread_id` is kept for API compatibility, but the current graph is stateless
      for streaming because `chat_node` uses `llm.invoke`.
    """
    print("stream_chatbot called with thread_id =", thread_id)
    
    # Stream text from the LLM. HuggingFace chat wrappers typically emit message-like chunks.
    for chunk in llm.stream(messages):
        delta = ""
        if hasattr(chunk, "content"):
            delta = chunk.content or ""
        else:
            delta = str(chunk)

        if delta:
            yield delta
