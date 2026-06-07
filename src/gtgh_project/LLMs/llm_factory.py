import os
import dotenv
from langchain_openai import ChatOpenAI
from src.gtgh_project.config import (LOCAL_AI_API_KEY, LOCAL_AI_ENDPOINT,
LOCAL_AI_MODEL, AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, AZURE_OPENAI_API_VERSION,
AZURE_OPENAI_DEPLOYMENT_NAME)
class LlmFactory:
    def __init__(self, localLlm, temperature):
        self.temperature = temperature
        if localLlm:
            config = {
                "api_key": LOCAL_AI_API_KEY,
                "base_url": LOCAL_AI_ENDPOINT,
                "model": LOCAL_AI_MODEL
            }
        else:
            config = {
                "api_key": AZURE_OPENAI_API_KEY,
                "base_url": AZURE_OPENAI_ENDPOINT,
                "model": AZURE_OPENAI_DEPLOYMENT_NAME
            }
        self.__create_llm(config)

    def get_llm(self):
        """Public method to get an LLM instance."""
        return self.llm

    def __create_llm(self, config):
        """Private method to create an LLM instance."""
        self.llm = ChatOpenAI(
            api_key=config["api_key"],
            base_url=config["base_url"],
            model=config["model"],
            temperature=self.temperature,
        ) 