import streamlit as st
import sys
from pathlib import Path


def _ensure_streamlit_runtime() -> None:
    if __name__ != "__main__":
        return

    from streamlit.runtime.scriptrunner_utils.script_run_context import get_script_run_ctx

    if get_script_run_ctx() is not None:
        return

    from streamlit.web import cli as stcli

    script_path = str(Path(__file__).resolve())
    sys.argv = ["streamlit", "run", script_path, *sys.argv[1:]]
    raise SystemExit(stcli.main())


_ensure_streamlit_runtime()

sys.path.insert(0, str(Path(__file__).parent))
from langgraph_backend import invoke_chatbot
from langchain_core.messages import HumanMessage

def render_app() -> None:
    st.title("LangGraph Chatbot")

    # Initialize message history
    if "messages_history" not in st.session_state:
        st.session_state["messages_history"] = []

    # Display previous messages
    for message in st.session_state["messages_history"]:
        with st.chat_message(message["role"]):
            st.text(message["content"])

    # Get user input
    user_input = st.chat_input("Type your message here...")
    if user_input:
        # Add user message to history
        st.session_state["messages_history"].append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.text(user_input)

        # Get chatbot response
        try:
            response = invoke_chatbot([HumanMessage(content=user_input)])
            ai_response = response["messages"][-1].content
            st.session_state["messages_history"].append({"role": "assistant", "content": ai_response})
            with st.chat_message("assistant"):
                st.text(ai_response)
        except Exception as e:
            st.error(f"Error getting response: {str(e)}")


render_app()