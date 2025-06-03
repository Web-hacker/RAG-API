import streamlit as st
import requests
import html
import time

st.set_page_config(page_title="RAG Agent", layout="wide")

st.sidebar.title("API Endpoints")
page = st.sidebar.radio(
    "Choose an option:",
    (" Chat with Chatbot", " Github Token", " Ingest Repo")
)

# -----------------------
# 1. Chat Interface
# -----------------------

if page == " Chat with Chatbot":
    st.title(" Chat with RAG Agent")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message("user" if msg["role"] == "user" else "assistant"):
            content = msg["content"]
            if "```" in content:
                blocks = content.split("```")
                for i, block in enumerate(blocks):
                    if i % 2 == 0:
                        st.markdown(block)
                    else:
                        st.code(block)
            else:
                st.markdown(content)
            if msg.get("sources"):
                with st.expander("📄 Sources"):
                    for s in msg["sources"]:
                        st.code(s)

    # Take user input
    user_query = st.chat_input("Ask something...")
    if user_query:
        # Show user message
        st.chat_message("user").markdown(user_query)
        st.session_state.messages.append({"role": "user", "content": user_query})

        # Call backend and render assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    res = requests.post("http://localhost:8000/query", json={"query": user_query})
                    res.raise_for_status()
                    data = res.json()
                    answer = data.get("answer", "No answer returned.")
                    sources = data.get("sources", [])
                    bot_message = st.empty()
                   
                    full_response = ""
                    for chunk in answer:
                       full_response += chunk
                       bot_message.markdown(full_response)
                       time.sleep(0.005)
        

                    if sources:
                        with st.expander("📄 Sources"):
                            for s in sources:
                                st.code(s)

                    # Store bot message in state
                    st.session_state.messages.append({
                        "role": "bot",
                        "content": answer,
                        "sources": sources
                    })

                except Exception as e:
                    st.error(f"⚠️ Error: {e}")
                    st.session_state.messages.append({
                        "role": "bot",
                        "content": f"⚠️ Error: {e}"
                    })

# -----------------------
# 2. Github Token Page 
# -----------------------
elif page == " Github Token":
    st.title(" Share your github token.")

    user_github_token = st.chat_input("Enter github token...")
    
    if user_github_token:
        with st.spinner("Uploading..."):
            try:
                response = requests.post("http://localhost:8000/token", json={"Github_token": user_github_token})
                st.success(response.json()["message"])
            except Exception as e:
                st.error(f"Upload failed: {e}")



# Button click
button = st.sidebar.button("Sync with latest version of Documentation.")

# Handle sync
if button:
    with st.spinner("Syncing..."):
        try:
            res = requests.post("http://localhost:8000/sync")
            print(res.json())
            st.sidebar.success("Synced successfully")
        except Exception as e:
            st.sidebar.error(f"Syncing failed: {e}")
    #st.rerun()

# Display message if present
#if "sync_message" in st.session_state:
#    msg_type, msg_text = st.session_state.sync_message
#    if msg_type == "success":
#        st.sidebar.success(msg_text)
#    else:
#        st.sidebar.error(msg_text)
#    
#    # Delay and clear the message
#    time.sleep(5)
#    del st.session_state.sync_message
#    st.rerun()

# -----------------------
# 3. Repo Clone
# -----------------------
elif page == " Ingest Repo":
    st.title(" Ingest GitHub Repo for Embedding in Vector Database.")

    repo_url = st.text_input("GitHub Repo URL")
    branch = st.text_input("Branch", value="main")

    if st.button("Ingest"):
        with st.spinner("Upserting..."):
            try:
                res = requests.post("http://localhost:8000/ingest", json={"repo_url": repo_url, "branch": branch})
                res.raise_for_status()
                st.success(res.json()["message"])
            except Exception as e:
                st.error(f"Clone failed: {e}")
