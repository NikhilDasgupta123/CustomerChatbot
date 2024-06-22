import streamlit as st 
from dotenv import load_dotenv
from langchain_community.utilities import SQLDatabase
from utils.config import Confg

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain.chat_models import AzureChatOpenAI
import os
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

# Sql Alc Initialize Database
def init_database():
    db = SQLDatabase.from_uri(f"mysql+mysqlconnector://{Confg.db_user}:{Confg.db_password}@{Confg.db_host}/{Confg.db_name}")
    return db

def llmmodel():
    model_name = os.getenv("AZURE_OPENAI_API_DEPLOYMENT_NAME")
    azure_openai_api_key = os.environ["AZURE_OPENAI_API_KEY"]
    azure_openai_endpoint = os.environ["AZURE_OPENAI_ENDPOINT"]

    llm = AzureChatOpenAI(
        openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        azure_deployment=model_name,
        model_name=model_name,
        temperature=0.7,  # Increase temperature for more diverse responses
        max_tokens=300,  # Increase max tokens for longer responses
        top_p=0.9,  # Use nucleus sampling for better diversity
    )

    return llm

def get_sql_chain(db, username):
    template = """
    You are a Customer Service Bot designed to assist {username}. You can help with inquiries related to our products, orders, and shipments.

    Based on our database schema provided below, write a SQL query that would answer the user's question. Take into account our conversation history.

    <SCHEMA>{schema}</SCHEMA>

    Conversation History: {chat_history}

    Please provide your SQL query without any additional text or formatting.

    For example:
    Question: What is the status of order ID 301?
    SQL Query: SELECT status FROM Orders WHERE order_id = 301;

    Your turn:

    Question: {question}
    SQL Query:
    """

    prompt = ChatPromptTemplate.from_template(template)

    def get_schema(_):
        return db.get_table_info()

    return (
        RunnablePassthrough.assign(schema=get_schema)
        | prompt
        | llmmodel()
        | StrOutputParser()
    )

def get_response(user_query: str, db: SQLDatabase, chat_history: list, username):
    sql_chain = get_sql_chain(db, username)

    template = """
    You are Customer Service Bot, assisting {username}. You specialize in handling inquiries regarding our products, orders, and shipments.

    Here is the context of our conversation history and database schema:

    <SCHEMA>{schema}</SCHEMA>

    Conversation History: {chat_history}
    SQL Query: <SQL>{query}</SQL>
    User question: {question}
    SQL Response: {response}

    Based on the information from the database, provide a detailed and human-friendly answer to the user's question. Ensure the response is descriptive and easily understandable.
    """

    prompt = ChatPromptTemplate.from_template(template)

    chain = (
        RunnablePassthrough.assign(query=sql_chain).assign(
            schema=lambda _: db.get_table_info(),
            response=lambda vars: db.run(vars["query"]),
        )
        | prompt
        | llmmodel()
        | StrOutputParser()
    )

    response = chain.invoke({
        "question": user_query,
        "chat_history": chat_history,
        "username": username,
    })
    
    return response

# main File
def main():
    db = init_database()

    with st.sidebar:
        st.subheader("Login")
        username = st.text_input("Name", value="", key="Name")

        if username:
            st.session_state.username = username

    # Check if username is provided
    if "username" in st.session_state:
        # session state
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = [
                AIMessage(content=f"""Hello {st.session_state.username}, I am Aarohaa's Customer Service Bot. I'm here to assist you with any questions or tasks you may have regarding our products, orders, or shipments. Feel free to ask!"""),
            ]

        # conversation
        for message in st.session_state.chat_history:
            if isinstance(message, AIMessage):
                with st.chat_message("AI"):
                    st.markdown(message.content)
            elif isinstance(message, HumanMessage):
                with st.chat_message("Human"):
                    st.markdown(message.content)

        user_query = st.chat_input("Type Something...")
        if user_query is not None and user_query.strip() != "":
            st.session_state.chat_history.append(HumanMessage(content=user_query))

            # Write Human message
            with st.chat_message("Human"):
                st.markdown(user_query)

            # Write AI Message
            with st.chat_message("AI"):
                response = get_response(user_query, db, st.session_state.chat_history, st.session_state.username)
                
                # Parse the response to generate a table
                if "Order ID" in response:
                    response_lines = response.split("\n")
                    order_details = [line.split(": ") for line in response_lines if ": " in line]
                    order_dict = {item[0]: item[1] for item in order_details}
                    
                    st.write(f"Sure, {st.session_state.username}! I have checked our database and found the following details about your order:")
                    st.table(order_dict)
                else:
                    st.write(response)

            st.session_state.chat_history.append(AIMessage(content=response))  
    else:
        st.write("Please enter your name in the sidebar to start the chat.")

if __name__ == '__main__':
    main()
