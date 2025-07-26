
import streamlit as st

st.set_page_config(page_title="Shuttle Debug App", layout="wide")

st.sidebar.title("🛠️ Debug Sidebar")
st.sidebar.write("If you see this, sidebar is working.")

st.markdown("## 🚌 Shuttle Debug App")
st.write("✅ App loaded successfully.")

if st.sidebar.button("Generate Sample Data"):
    st.success("🎉 Sample data generation triggered (mock)")
else:
    st.info("🔍 Waiting for user to click button...")
