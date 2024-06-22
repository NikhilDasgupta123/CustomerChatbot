import warnings
warnings.filterwarnings('ignore')

import os
import openai
from openai import AzureOpenAI
from dotenv import load_dotenv

load_dotenv()

# openai.api_key = os.getenv("OPENAI_API_KEY")
# openai.api_base = os.getenv("AZURE_OPENAI_ENDPOINT")
# openai.api_type = os.getenv("OPENAI_API_TYPE")
# openai.api_version = os.getenv("OPENAI_API_VERSION")

# client = AzureOpenAI(
#     api_key=os.getenv("AZURE_OPENAI_API_KEY"),  
#     api_version="AZURE_OPENAI_API_VERSION",
#     azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
# )


#client = AzureOpenAI(azure_endpoint="https://oai-ecirkle-dev-01.openai.azure.com/", api_key = "e125adf912704e61afef50706b6f2f1f", api_version = "2024-02-01" )
client = AzureOpenAI(azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"), api_key = os.getenv("AZURE_OPENAI_API_KEY"), api_version = os.getenv("AZURE_OPENAI_API_VERSION") )
print(client)

# openai.api_key = os.getenv("OPENAI_API_KEY")
# openai.azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
# openai.api_type = os.getenv("OPENAI_API_TYPE")
# openai.api_version = os.getenv("OPENAI_API_VERSION")

print("am now here ", client.api_key)

def chat_with_gpt(prompt):
    response = openai.chat.completions.create(
        model = "gpt-3.5-turbo",
        messages = [{"role": "user", "content":prompt}],
        max_tokens = 7
    )

    return response.choices[0].message.content.strip()

if __name__ == "__main__":
    while True:
        user_input = input("You : ")
        if(user_input.lower() in ["quit", "bye", "exit"]):
            break

        response = chat_with_gpt(user_input)
        print("Chatbot: ", response)


        