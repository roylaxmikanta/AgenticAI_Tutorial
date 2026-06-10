# TODO

## Task: Add reliable streaming for LangGraph chatbot UI

- [ ] Inspect current code (done): `streamlit_frontend.py`, `streamlit_frontend_streaming.py`, `langgraph_backend.py`
- [ ] Fix backend `stream_chatbot()` so it yields only string deltas and handles chunks safely
- [ ] Update `streamlit_frontend.py` to stream response correctly and store full assistant reply in session history
- [ ] (Optional) Make `streamlit_frontend_streaming.py` use the same backend streaming path (if possible with LangGraph semantics)
- [ ] Run `streamlit` locally to verify streaming output

