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
      temperature=0.0)
  
  return llm




def get_sql_chain(db):
  template = """
  You are Customer Service Bot aims to automate customer care interactions by accessing and retrieving information from various tables in the database. 
  The bot will handle queries related to order status, product details, and shipment status. 
  Based on the table schema below, write a SQL query that would answer the user's question. 
  Take the conversation history into account.

  <SCHEMA>{schema}</SCHEMA>

  Conversation History: {chat_history}

  Write only the SQL query and nothing else. Do not wrap the SQL query in any other text, not even backticks.

  For example:
  Question: What is the status of order ID 301?
  SQL Query: SELECT status FROM Orders WHERE order_id = 301;
  Question: Provide details of product ID 203.
  SQL Query: SELECT * FROM Products WHERE product_id = 203;
  Question: When was order ID 307 shipped?
  SQL Query: SELECT shipment_date FROM Shipments WHERE order_id = 307;

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









def get_response(user_query: str, db: SQLDatabase, chat_history: list):
  sql_chain = get_sql_chain(db)
  
  template = """
    You are Customer Service Bot aims to automate customer care interactions by accessing and retrieving information from various tables in the database. 
    The bot will handle queries related to order status, product details, and shipment status.
    Based on the table schema below, question, sql query, and sql response, write a natural language response.
    <SCHEMA>{schema}</SCHEMA>

    Conversation History: {chat_history}
    SQL Query: <SQL>{query}</SQL>
    User question: {question}
    SQL Response: {response}"""
  
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
  
  return chain.invoke({
    "question": user_query,
    "chat_history": chat_history,
  })






# main File
def main():

  if init_database():
    # with st.sidebar:
    #   st.subheader("Login")
    
    #   st.text_input("Name", value="", key="Name")
       
  



    # session state
    if "chat_history" not in st.session_state:
      st.session_state.chat_history = [
          AIMessage(content="""Hello, I am a Aarohaa Service bot. I'm here to assist you with any questions or tasks you may have. 
                    Whether you need information, guidance, or support, feel free to ask. How can I help you today?"""),
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

      # Write Ai Message
      with st.chat_message("AI"):
        response = get_response(user_query, 
                                SQLDatabase.from_uri(f"mysql+mysqlconnector://{Confg.db_user}:{Confg.db_password}@{Confg.db_host}/{Confg.db_name}"), 
                                st.session_state.chat_history)
        st.write(response)

      st.session_state.chat_history.append(AIMessage(content=response))  




if __name__ == '__main__':
    main()
