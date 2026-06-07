from src.gtgh_project.LLMs.llm_factory import LlmFactory
from src.gtgh_project.LLMs.system_prompt import get_system_prompt
from langchain_core.output_parsers import StrOutputParser


class RagChain:
    def __init__(self, local_llm=True):
        self.output_parser = StrOutputParser()
        self.chain = (
            get_system_prompt()
            | LlmFactory(localLlm=local_llm).get_llm()
            | self.output_parser
        )

    def invoke(self, question: str, context: str):
        return self.chain.invoke(
            {
                "question": question,
                "context": context,
            }
        )  