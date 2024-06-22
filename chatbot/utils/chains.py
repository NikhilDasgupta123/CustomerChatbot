from langchain_core.prompts import ChatPromptTemplate
from langchain_community.utilities import SQLDatabase
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from utils.llmModel import loadModel



# Create SQL Chain, Few Shot Prompt
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
        | loadModel()
        | StrOutputParser()
    )




# Generate Response, Chat History
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

    Based on the information, provide a detailed and human-friendly answer to the user's question. Ensure the response is descriptive and easily understandable.
    """

    prompt = ChatPromptTemplate.from_template(template)

    chain = (
        RunnablePassthrough.assign(query=sql_chain).assign(
            schema=lambda _: db.get_table_info(),
            response=lambda vars: db.run(vars["query"]),
        )
        | prompt
        | loadModel()
        | StrOutputParser()
    )

    response = chain.invoke({
        "question": user_query,
        "chat_history": chat_history,
        "username": username,
    })
    
    return response

