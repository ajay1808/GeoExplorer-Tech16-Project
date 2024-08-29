import streamlit as st
import requests
import pickle
from llama_index.core import VectorStoreIndex, Document
from llama_index.llms.openai import OpenAI
from llama_index.core.tools import FunctionTool
from llama_index.core.agent import ReActAgent
import os

# Streamlit Title
st.title("Geo Explorer β 🌍")

# Introduction Section
st.markdown("""
**Discover the details behind any address with just a few clicks.**

Enter an address to start. The app will use the HERE API to gather geographic information about the location. Then, simply ask a question—whether it's about nearby spots or neighborhood characteristics—and our Agent will fetch the relevant details for you.

With built-in memory, Geo Explorer β keeps track of your queries, making your exploration smoother and more intuitive.

---

Before you get started, you'll need an OpenAI API key. If you don't have one, you can sign up [here](https://platform.openai.com/api-keys).

""")


# Initialize Session State for messages, agent, and memory
if "messages" not in st.session_state:
    st.session_state.messages = []

if "agent" not in st.session_state:
    st.session_state.agent = None

if "memory" not in st.session_state:
    st.session_state.memory = []

# Input for API Keys and Address
OPEN_AI_API_KEY = st.text_input("Enter OpenAI API Key", type="password")
address = st.text_input("Enter an Address")

# Load or create the vector index
if OPEN_AI_API_KEY and address:
    HERE_API_KEY = st.secrets["HERE_API"]

    os.environ["OPENAI_API_KEY"] = OPEN_AI_API_KEY

    # Tool Functions
    def get_geocode_data_tool(address: str) -> dict:
        url = f"https://geocode.search.hereapi.com/v1/geocode?q={address}&apiKey={HERE_API_KEY}"
        response = requests.get(url)
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": "Error fetching data from HERE API."}

    def discover_nearby_tool(lat: float, lng: float, query: str, lang: str = 'en') -> dict:
        params = {
            'at': f"{lat},{lng}",
            'q': query,
            'limit': 5,
            'lang': lang,
            'apiKey': HERE_API_KEY
        }
        url = f"https://discover.search.hereapi.com/v1/discover"
        response = requests.get(url, params=params)
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"Error: {response.status_code}"}

    geocoding = FunctionTool.from_defaults(fn=get_geocode_data_tool)
    discover_nearby_places = FunctionTool.from_defaults(fn=discover_nearby_tool)

    # Initialize LLM and ReAct agent with memory
    llm = OpenAI(model="gpt-4o-mini", temperature = 0.2)
    agent = ReActAgent.from_tools([geocoding, discover_nearby_places], llm=llm, verbose=True)
    
    st.session_state.agent = agent

# Chat-like interface with memory
if st.session_state.agent:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask about your neighborhood..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Agent memory retrieval
        memory_input = "\n".join([f"User: {msg['content']}" for msg in st.session_state.messages if msg['role'] == 'user'])
        full_prompt = (
            "You are an assistant. Always use the tools provided to answer the question. "
            "If the tools do not give you the answer, respond that you do not have the information to answer the question.\n\n"
            f"Memory:\n{memory_input}\n\n"
            f"This is the given address: {address}, answer this question: {prompt}"
        )
        
        with st.chat_message("assistant"):
            result = st.session_state.agent.chat(full_prompt)
            st.markdown(result)
            st.session_state.messages.append({"role": "assistant", "content": result})
            st.session_state.memory.append(result)  
