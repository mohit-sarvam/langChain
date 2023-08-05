from langchain import PromptTemplate

import langchain_visualizer
import asyncio
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.chains import LLMChain, RetrievalQA

from third_parties.databricks import get_full_documentation

if __name__ == "__main__":
    print("Hello LangChain!")

    template = """
         Given my use case {use_case}, I want you to find appropriate Databricks API and create:
         1. Short summary
         2. How to use this API
     """

    prompt_template = PromptTemplate(input_variables=["use_case"], template=template)


    text_splitter = CharacterTextSplitter(
        chunk_size=200, chunk_overlap=30, separator="\n"
    )
    docs = text_splitter.split_documents(documents=get_full_documentation())
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local("faiss_index_react")
    new_vectorstore = FAISS.load_local("faiss_index_react", embeddings)

    qa = RetrievalQA.from_chain_type(
        llm=OpenAI(), chain_type="stuff", retriever=new_vectorstore.as_retriever()
    )

    async def async_run_agent():
        return qa.run(
            prompt_template.format_prompt(
                use_case="Submit Sql queries to my lakehouse."
            ).to_string()
        )

    langchain_visualizer.visualize(async_run_agent)
