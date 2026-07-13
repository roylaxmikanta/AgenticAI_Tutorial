import streamlit as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from langgraph_backend import stream_chatbot
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage

st.title("LangGraph Chatbot (Streaming)")

# Initialize message history
if "messages_history" not in st.session_state:
    # Each item: {"role": "user"|"assistant", "content": str}
    st.session_state["messages_history"] = []


def history_to_langchain_messages(history: list[dict]) -> list[BaseMessage]:
    """Convert our session history into LangChain BaseMessage objects."""
    messages: list[BaseMessage] = []
    for msg in history:
        role = msg.get("role")
        content = msg.get("content", "")
        if role == "user":
            messages.append(HumanMessage(content=content))
        elif role == "assistant":
            messages.append(AIMessage(content=content))
    return messages


# Display previous messages
for message in st.session_state["messages_history"]:
    with st.chat_message(message["role"]):
        st.markdown(message.get("content", ""))

# Get user input
user_input = st.chat_input("Type your message here...")
if user_input:
    st.session_state["messages_history"].append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.markdown(user_input)

    # Get chatbot response with streaming
    full_response = ""
    try:
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            prior_messages = history_to_langchain_messages(st.session_state["messages_history"])

            with st.spinner("Thinking..."):
                for delta in stream_chatbot(prior_messages):
                    # stream_chatbot yields string deltas
                    if not isinstance(delta, str):
                        delta = str(delta)
                    full_response += delta
                    message_placeholder.markdown(full_response + "▌")

            # Final render (without cursor)
            message_placeholder.markdown(full_response or " ")

        st.session_state["messages_history"].append({"role": "assistant", "content": full_response})

    except Exception as e:
        st.error(f"Error getting response: {str(e)}")
        # Keep history consistent even on failure
        st.session_state["messages_history"].append({"role": "assistant", "content": f"[Error] {str(e)}"})
