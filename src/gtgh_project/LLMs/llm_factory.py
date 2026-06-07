from langchain_openai import ChatOpenAI, AzureChatOpenAI
from src.gtgh_project.config import (LOCAL_AI_API_KEY, LOCAL_AI_ENDPOINT,
LOCAL_AI_MODEL, AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY,
AZURE_OPENAI_DEPLOYMENT_NAME, AZURE_OPENAI_API_VERSION)

class LlmFactory:
    def __init__(self, local_llm, temperature):
        self.__create_llm(local_llm, temperature)

    def get_llm(self):
        """Public method to get an LLM instance."""
        return self.llm

    def __create_llm(self, local, temperature):
        """Private method to create an LLM instance."""
        if local:
            self.llm = ChatOpenAI(
            api_key = LOCAL_AI_API_KEY,
            base_url = LOCAL_AI_ENDPOINT,
            model = LOCAL_AI_MODEL,
            temperature = temperature
            )
        else:
            self.llm = AzureChatOpenAI(
            openai_api_key = AZURE_OPENAI_API_KEY,
            azure_endpoint = AZURE_OPENAI_ENDPOINT,
            azure_deployment= AZURE_OPENAI_DEPLOYMENT_NAME,
            openai_api_version = AZURE_OPENAI_API_VERSION,
            temperature=temperature
            )