import os
import dotenv
from langchain_openai import ChatOpenAI

class LlmFactory:
    def __init__(self, localLlm=True, temperature=0.7):
        self.temperature = temperature
        dotenv.load_dotenv() 
        if(localLlm):
            config = {
                "api_key": os.getenv("AI_API_KEY", "my-secret-key"),
                "base_url": os.getenv("AI_ENDPOINT", "http://localhost:4000"),
                "model": os.getenv("AI_MODEL", "gpt-4-turbo")
            }
        else:
            config = {
                "api_key": os.getenv("GROQ_API_KEY"),
                "base_url": os.getenv("GROQ_ENDPOINT", "https://api.groq.com/openai/v1"),
                "model": os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
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