from dotenv import load_dotenv
import os

load_dotenv()


class Confg:
    # Database Config
    db_user = "root"
    db_password = "1234"
    db_host = "localhost"
    db_name = "chatbot"

    # Model Config
    model_name = os.getenv("AZURE_OPENAI_API_DEPLOYMENT_NAME")
    azure_openai_api_key = os.environ["AZURE_OPENAI_API_KEY"]
    azure_openai_endpoint = os.environ["AZURE_OPENAI_ENDPOINT"]