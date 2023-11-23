import streamlit as st
from llama_index.llms import OpenAI
from llama_index import VectorStoreIndex, ServiceContext, SimpleDirectoryReader
import openai

openai.api_key = st.secrets["openaikey"]
st.header("Chat directly with the LNU Student Handbook 💬 📚!")

if "messages" not in st.session_state.keys():
    st.session_state.messages = [
        {"role": "assistant", "content": "Ask me a question about Leyte Normal University's student handbook!"}
    ]


@st.cache_resource(show_spinner=False)
def load_data():
    with st.spinner(text="Loading and indexing the Student Handbook – hang tight! This should take 1-3 minutes."):
        reader = SimpleDirectoryReader(
            input_files=["./handbook_data.json"]
        )
        docs = reader.load_data()
        service_context = ServiceContext.from_defaults(llm=OpenAI(model="gpt-3.5-turbo", temperature=0.5,
                                                                  system_prompt="You are an expert on the Leyte Normal University Student Handbook and your job is to answer technical questions. Assume that all questions are related to the Leyte Normal University Student Handbook. Keep your answers technical and based on facts – do not hallucinate features."))
        temp_index = VectorStoreIndex.from_documents(docs, service_context=service_context)
        return temp_index


index = load_data()

chat_engine = index.as_chat_engine(chat_mode="context", verbose=True)

if prompt := st.chat_input("Your question"):
    st.session_state.messages.append({"role": "user", "content": prompt})

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = chat_engine.chat(prompt)
            st.write(response.response)
            message = {"role": "assistant", "content": response.response}
            st.session_state.messages.append(message)
