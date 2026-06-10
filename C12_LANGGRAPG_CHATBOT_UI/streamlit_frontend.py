import streamlit as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from langgraph_backend import invoke_chatbot, stream_chatbot
from langchain_core.messages import HumanMessage

st.title("LangGraph Chatbot")

# Initialize message history
if "messages_history" not in st.session_state:
    st.session_state["messages_history"] = []

# Display previous messages
for message in st.session_state["messages_history"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Get user input
user_input = st.chat_input("Type your message here...")
if user_input:
    # Add user message to history
    st.session_state["messages_history"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Get chatbot response with streaming
    try:
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""

            # Build messages from history for better multi-turn context.
            prior_messages = []
            for msg in st.session_state["messages_history"]:
                if msg["role"] == "user":
                    prior_messages.append(HumanMessage(content=msg["content"]))
                else:
                    # Assistant messages are stored as plain text; keep them as AIMessage via BaseMessage interface.
                    from langchain_core.messages import AIMessage
                    prior_messages.append(AIMessage(content=msg["content"]))

            # Stream response
            for delta in stream_chatbot(prior_messages + [HumanMessage(content=user_input)]):
                full_response += delta
                message_placeholder.markdown(full_response + "▌")

            message_placeholder.markdown(full_response)

        st.session_state["messages_history"].append({"role": "assistant", "content": full_response})
    except Exception as e:
        st.error(f"Error getting response: {str(e)}")