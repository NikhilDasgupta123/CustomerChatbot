import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.utilities import SQLDatabase


from utils.db import mySQL
from utils.chains import get_response
from utils.config import Confg



## Streamlit UI
def main():
    db = mySQL()

    with st.sidebar:
        st.subheader("Login")
        username = st.text_input("Name", value="", key="Name")

        if username:
            st.session_state.username = username

    # Check if username is provided
    if "username" in st.session_state:
        # Initialize chat history
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = [
                AIMessage(content=f"""Hello {st.session_state.username}, I am Aarohaa's Customer Service Bot. I'm here to assist you with any questions or tasks you may have regarding our products, orders, or shipments. Feel free to ask!"""),
            ]

        # Display conversation history
        for message in st.session_state.chat_history:
            if isinstance(message, AIMessage):
                with st.chat_message("AI"):
                    st.markdown(message.content)
            elif isinstance(message, HumanMessage):
                with st.chat_message("Human"):
                    st.markdown(message.content)

        # Input field for user's query
        user_query = st.chat_input("Type Something...")
        if user_query is not None and user_query.strip() != "":
            st.session_state.chat_history.append(HumanMessage(content=user_query))

            # Write Human message
            with st.chat_message("Human"):
                st.markdown(user_query)

            # Write AI Message
            with st.chat_message("AI"):
                response = get_response(user_query, db, st.session_state.chat_history, st.session_state.username)
                
                st.write(response)

            st.session_state.chat_history.append(AIMessage(content=response))

        
    else:
        st.write("Please enter your name in the sidebar to start the chat.")

if __name__ == '__main__':
    main()