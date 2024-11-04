"""
Streamlit Integration Example
===========================
Example of using AI Orchestrator with Streamlit.
"""

import streamlit as st
import asyncio
from ai_orchestrator import AIOrchestrator

# Initialize orchestrator
@st.cache_resource
def get_orchestrator():
    return AIOrchestrator(api_key=st.secrets["api_key"])

orchestrator = get_orchestrator()

# App interface
st.title("AI Orchestrator Demo")

# Session management
if "session_id" not in st.session_state:
    st.session_state.session_id = "streamlit-session"

# User input
user_input = st.text_area("Enter your text:")
context = st.text_input("Context (optional):", "")

if st.button("Process"):
    if user_input:
        with st.spinner("Processing..."):
            # Process input
            result = asyncio.run(orchestrator.process_input(
                session_id=st.session_state.session_id,
                user_input=user_input,
                context_updates={"context": context} if context else None
            ))
            
            # Display results
            st.json(result)
    else:
        st.warning("Please enter some text to process")

# Session info
if st.button("Show Session Info"):
    info = orchestrator.get_session_info(st.session_state.session_id)
    st.json(info)