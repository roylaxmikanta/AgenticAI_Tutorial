import streamlit as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from C15_LangGraph_SQLite_Database_Integration.langgraph_database_backend import chatbot, invoke_chatbot, retrieve_all_threads, stream_chatbot

from langchain_core.messages import HumanMessage, AIMessage
import uuid

def generate_thread_id():
    return str(uuid.uuid4())

def reset_chat():
    thread_id = generate_thread_id()
    st.session_state['thread_id'] = thread_id
    if thread_id not in st.session_state['chat_threads']:
        st.session_state['chat_threads'].append(thread_id)
    st.session_state['thread_names'][thread_id] = "New Chat"
    st.session_state['message_history'] = []

def load_conversation(thread_id):
    state = chatbot.get_state(config={'configurable': {'thread_id': thread_id}})
    return state.values.get('messages', [])

# Session Setup
if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []

if 'thread_id' not in st.session_state:
    st.session_state['thread_id'] = generate_thread_id()

if 'chat_threads' not in st.session_state:
    st.session_state['chat_threads'] = []

if 'thread_names' not in st.session_state:
    st.session_state['thread_names'] = {}

# Load existing threads from database (all strings)
existing_threads = retrieve_all_threads()
for tid in existing_threads:
    if tid not in st.session_state['chat_threads']:
        st.session_state['chat_threads'].append(tid)
    if tid not in st.session_state['thread_names']:
        st.session_state['thread_names'][tid] = f"Chat {tid}"

if st.session_state['thread_id'] not in st.session_state['chat_threads']:
    st.session_state['chat_threads'].append(st.session_state['thread_id'])

# Sidebar UI
st.sidebar.title('LangGraph Chatbot (SQLite)')

if st.sidebar.button('New Chat'):
    reset_chat()

st.sidebar.header('My Conversations')

for thread_id in st.session_state['chat_threads'][::-1]:
    thread_name = st.session_state['thread_names'].get(thread_id, "Untitled")
    if st.sidebar.button(thread_name, key=thread_id):
        st.session_state['thread_id'] = thread_id
        messages = load_conversation(thread_id)
        
        temp_messages = []
        for msg in messages:
            if isinstance(msg, HumanMessage):
                role = 'user'
            else:
                role = 'assistant'
            temp_messages.append({'role': role, 'content': msg.content})
        
        st.session_state['message_history'] = temp_messages

# Main UI
for message in st.session_state['message_history']:
    with st.chat_message(message['role']):
        st.markdown(message['content'])

user_input = st.chat_input('Type here...')

if user_input:
    thread_key = st.session_state['thread_id']
    if thread_key not in st.session_state['thread_names'] or st.session_state['thread_names'][thread_key] == "New Chat":
        conv_name = user_input[:50] + ("..." if len(user_input) > 50 else "")
        st.session_state['thread_names'][thread_key] = conv_name

    st.session_state['message_history'].append({'role': 'user', 'content': user_input})
    with st.chat_message('user'):
        st.markdown(user_input)

    CONFIG = {'configurable': {'thread_id': st.session_state['thread_id']}}

    ai_message_content = ""
    with st.chat_message("assistant"):
        try:
            for text in stream_chatbot([HumanMessage(content=user_input)], thread_id=st.session_state['thread_id']):
                ai_message_content += text
            st.markdown(ai_message_content)
        except Exception as e:
            st.error(f"Error: {str(e)}")
            ai_message_content = f"[Error] {str(e)}"

    st.session_state['message_history'].append({'role': 'assistant', 'content': ai_message_content})