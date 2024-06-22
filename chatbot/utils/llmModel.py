from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os
from langchain.chat_models import AzureChatOpenAI
from utils.config import Confg
load_dotenv()


# Load LLM Model
def loadModel():
    model_name = os.getenv("AZURE_OPENAI_API_DEPLOYMENT_NAME")
    azure_openai_api_key = os.environ["AZURE_OPENAI_API_KEY"]
    azure_openai_endpoint = os.environ["AZURE_OPENAI_ENDPOINT"]

    llm = AzureChatOpenAI(
        openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        azure_deployment=model_name,
        model_name=model_name,
        temperature=0.7,
        max_tokens=300,
        top_p=0.9,
    )

    return llm