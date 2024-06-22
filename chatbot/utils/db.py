from langchain_community.utilities import SQLDatabase
from .config import Confg


    
    
## Databases Connection
def mySQL():
    db_uri = SQLDatabase.from_uri(f"mysql+mysqlconnector://{Confg.db_user}:{Confg.db_password}@{Confg.db_host}/{Confg.db_name}")
    return db_uri
    

