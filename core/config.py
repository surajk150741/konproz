import os
from dotenv import load_dotenv
load_dotenv()
OPENAI_API_KEY=os.getenv("OPENAI_API_KEY_KONPROZ")

class Config:
    PLAYBOOK_FILE="document/GST_Smart_Guide.pdf"

setting=Config()
# print(setting.CHAT_MEMORY_DATABASE)
