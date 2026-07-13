import streamlit as st
# from langgraph_tool_backend import chatbot
from langgraph_backend import chatbot
from langchain_core.messages import HumanMessage, AIMessage
import uuid

# **************************************** utility functions *************************

def generate_thread_id():
    thread_id = uuid.uuid4()
    return thread_id

def reset_chat():
    thread_id = generate_thread_id()
    st.session_state['thread_id'] = thread_id
    add_thread(st.session_state['thread_id'], "New Chat")
    st.session_state['message_history'] = []

def add_thread(thread_id, name=None):
    if thread_id not in st.session_state['chat_threads']:
        st.session_state['chat_threads'].append(thread_id)
    if 'thread_names' not in st.session_state:
        st.session_state['thread_names'] = {}
    if name:
        st.session_state['thread_names'][str(thread_id)] = name

def load_conversation(thread_id):
    state = chatbot.get_state(config={'configurable': {'thread_id': thread_id}})
    # Check if messages key exists in state values, return empty list if not
    return state.values.get('messages', [])


# **************************************** Session Setup ******************************
if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []

if 'thread_id' not in st.session_state:
    st.session_state['thread_id'] = generate_thread_id()

if 'chat_threads' not in st.session_state:
    st.session_state['chat_threads'] = []

if 'thread_names' not in st.session_state:
    st.session_state['thread_names'] = {}

add_thread(st.session_state['thread_id'])


# **************************************** Sidebar UI *********************************

st.sidebar.title('LangGraph Chatbot')

if st.sidebar.button('New Chat'):
    reset_chat()

st.sidebar.header('My Conversations')

for thread_id in st.session_state['chat_threads'][::-1]:
    thread_name = st.session_state['thread_names'].get(str(thread_id), "Untitled")
    if st.sidebar.button(thread_name, key=str(thread_id)):
        st.session_state['thread_id'] = thread_id
        messages = load_conversation(thread_id)

        temp_messages = []

        for msg in messages:
            if isinstance(msg, HumanMessage):
                role='user'
            else:
                role='assistant'
            temp_messages.append({'role': role, 'content': msg.content})

        st.session_state['message_history'] = temp_messages


# **************************************** Main UI ************************************

# loading the conversation history
for message in st.session_state['message_history']:
    with st.chat_message(message['role']):
        st.text(message['content'])

user_input = st.chat_input('Type here')

if user_input:

    # Set conversation name based on first message if not already set
    thread_key = str(st.session_state['thread_id'])
    if thread_key not in st.session_state['thread_names'] or st.session_state['thread_names'][thread_key] == "New Chat":
        # Use first 50 characters of the user message as the conversation name
        conv_name = user_input[:50] + ("..." if len(user_input) > 50 else "")
        st.session_state['thread_names'][thread_key] = conv_name

    # first add the message to message_history
    st.session_state['message_history'].append({'role': 'user', 'content': user_input})
    with st.chat_message('user'):
        st.text(user_input)

    CONFIG = {'configurable': {'thread_id': st.session_state['thread_id']}}

    ai_message_content_list = []
    with st.chat_message("assistant"):
        def ai_only_stream():
            for message_chunk, metadata in chatbot.stream(
                {"messages": [HumanMessage(content=user_input)]},
                config=CONFIG,
                stream_mode="messages"
            ):
                if isinstance(message_chunk, AIMessage):
                    chunk = message_chunk.content
                    ai_message_content_list.append(chunk)
                    yield chunk

        st.write_stream(ai_only_stream())

    ai_message_content = "".join(ai_message_content_list)
    st.session_state['message_history'].append({'role': 'assistant', 'content': ai_message_content})
    st.rerun()


    ai_message_content = ""
    with st.chat_message("assistant"):
        def ai_only_stream():
            for message_chunk, metadata in chatbot.stream(
                {"messages": [HumanMessage(content=user_input)]},
                config=CONFIG,
                stream_mode="messages"
            ):
                if isinstance(message_chunk, AIMessage):
                    # yield only assistant tokens
                    yield message_chunk.content

        for chunk in ai_only_stream():
            ai_message_content += chunk
        st.write(ai_message_content)

    st.session_state['message_history'].append({'role': 'assistant', 'content': ai_message_content})