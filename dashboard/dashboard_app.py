import streamlit as st
import sqlite3
import pandas as pd
from src.memory_server import DB_PATH, add_memory, get_memories, list_people, consolidate_memories

st.set_page_config(page_title="Reachy Dashboard", layout="wide")

st.title("🤖 Reachy Mini Dashboard")

# Memories Section
st.header("🧠 Memories")

col1, col2 = st.columns(2)

with col1:
    st.subheader("People")
    # Refresh list
    people = list_people.fn()
    selected_person = st.selectbox("Select Person", ["ROBOT"] + people)

with col2:
    st.subheader("Actions")
    if st.button("Consolidate Memories (Run Nightly Task)"):
        with st.spinner("Consolidating..."):
            result = consolidate_memories.fn()
            st.success(result)

# View Memories
st.subheader(f"Memories for {selected_person}")

# Fetch data directly from DB for better table view, or use get_memories
conn = sqlite3.connect(DB_PATH)
query = f"SELECT id, timestamp, memory_type, text FROM memories WHERE entity_name = '{selected_person}' ORDER BY timestamp DESC"
df = pd.read_sql_query(query, conn)
conn.close()

st.dataframe(df, use_container_width=True)

# Add Memory
st.subheader("Add New Memory")
with st.form("new_memory"):
    new_mem_text = st.text_area("Memory Text")
    submitted = st.form_submit_button("Add Memory")
    if submitted and new_mem_text:
        add_memory.fn(selected_person, new_mem_text)
        st.rerun()

# Operations
st.header("🔧 Robot Operations")
st.info("Robot status and controls will appear here when connected.")

# Camera Feed (Placeholder)
st.subheader("Live View")
st.image("https://via.placeholder.com/640x480?text=Camera+Feed+Placeholder", caption="Robot View")
