import streamlit as st
from openai import OpenAI
import os
from dotenv import load_dotenv
from datetime import datetime
from pathlib import Path
import requests
import chromadb
import json

BASE_DIR = Path(__file__).resolve().parent
ENV_PATH = BASE_DIR / ".env"

if ENV_PATH.exists():
    load_dotenv(dotenv_path=str(ENV_PATH), override=True)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    st.error(" OPENAI_API_KEY not found in .env file")
    st.stop()

client = OpenAI(api_key=OPENAI_API_KEY)


EXPRESS_API_BASE_URL = "http://127.0.0.1:8080/chat/request"
TIMEOUT = 15

CHROMA = str(BASE_DIR / "chroma_db")
collection_name = "chatbot_collection"
index_state = BASE_DIR / "index_state.json"
chroma = chromadb.PersistentClient(path=CHROMA)
collection = chroma.get_or_create_collection(name=collection_name)

def chunking_text(text, chunk_size=500, overlap=50):
   
    chunks = []
    start = 0
    text_length = len(text)
    
    while start < text_length:
        end = min(start + chunk_size, text_length)
        chunks.append(text[start:end])
        start += chunk_size - overlap
    
    return chunks

def embed(text):
  
    response = client.embeddings.create(
        input=text,
        model="text-embedding-3-large"
    )
    return response.data[0].embedding
def embedding_text(text_chunks):
 
    return [embed(chunk) for chunk in text_chunks]

def index()->dict:
    if index_state.exists():
        try:
            with open(index_state, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            return {}
    return {}

def save_index(state: dict):
    try:
        with open(index_state, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=4)
    except Exception as e:
        pass
    
def auto_index_start():
    files = sorted(BASE_DIR.glob("chat_history_*.txt"))
    if not files:
        return "No chat history files found for indexing."
    
    read_index_before = index()

    added_files = 0
    added_chunks = 0
    
    for file in files:
        if read_index_before.get(str(file)):
            continue
    
        text = file.read_text(encoding="utf-8", errors="ignore")
    
        text_chunks = chunking_text(text)
    
        ids, docs, embeddings, metas = [], [], [], []
    
        for i, ch in enumerate(text_chunks):
            ids.append(f"{file.stem}_chunk_{i}")
            docs.append(ch)
            embeddings.append(embed(ch))
            metas.append({"source": str(file), "chunk_index": i})
        
        collection.add(ids=ids, documents=docs, embeddings=embeddings, metadatas=metas)

        read_index_before[file.name] = "indexed"
        added_files += 1
        added_chunks += len(text_chunks)
        save_index(read_index_before)

    if added_files == 0:
        return " RAG ready (no new logs)."
    return f" RAG ready: {added_files} new log(s), {added_chunks} chunks."
         
if "rag_boot" not in st.session_state:
    st.session_state.rag_boot = True
    st.sidebar.info(auto_index_start())
    """her seferinde yeniden indeksleme yapmaması için oturum durumunu kontrol et"""
    
def retrieve(query: str, top_k=3):
    query_embedding = embed(query)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )

    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]

    return list(zip(docs, metas))
def build_context(hits, max_characters=1500):
    parts = []
    total_length = 0

    for doc, meta in hits:
        source = meta.get("source", "unknown")

        block = f"Source: {source}\n{doc}\n"
        if total_length + len(block) > max_characters:
            break

        parts.append(block)
        total_length += len(block)

    return "\n---\n".join(parts)

def save_chat_to_file(messages):
    """Save chat history to a timestamped file"""
    if not messages:
        return None
    
    time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"chat_history_{time_str}.txt"
    filepath = BASE_DIR / filename
    
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(f"Chat History - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        for msg in messages:
            f.write(f"{msg['role'].upper()}:\n{msg['content']}\n\n")
        f.write(f"Total messages: {len(messages)}\n")
    
    return filename


st.set_page_config(page_title="AI ChatBot", page_icon="🤖", layout="centered")

st.title("🤖 AI ChatBot")
st.markdown("---")

if "messages" not in st.session_state:
    st.session_state.messages = []


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Type your message here..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                hits = retrieve(prompt, top_k=3)
                context = build_context(hits)

                response = client.chat.completions.create(
                    model="gpt-4",
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a helpful assistant. Use CONTEXT from saved chat logs when relevant. If CONTEXT is empty/irrelevant, say you couldn't find it in saved logs."
                        },
                        {"role": "user", "content": f"CONTEXT:\n{context}\n\nQUESTION:\n{prompt}"},
                        *[{"role": m["role"], "content": m["content"]} for m in st.session_state.messages[-6:]],
                    ],
                    temperature=0.7,
                    max_tokens=500,
                )
                
                assistant_response = response.choices[0].message.content
                st.markdown(assistant_response)
                
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": assistant_response
                })
                
            except Exception as e:
                error_msg = f"Error: {str(e)}"
                st.markdown(error_msg)
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": error_msg
                })

with st.sidebar:
    st.header("Chat Controls")
    
    if st.button("Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    
    if st.button("Save Chat History", use_container_width=True):
        if st.session_state.messages:
            filename = save_chat_to_file(st.session_state.messages)
            if filename:
                st.success(f"Saved to {filename}")
        else:
            st.warning("No messages to save")
    
    st.markdown("---")
    st.caption(f"Messages: {len(st.session_state.messages)}")

