from langchain import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain


if __name__ == "__main__":
    print("Hello LangChain!")

    template = """
         Given my use case {use_case}, I want you to find appropriate Databricks API and create:
         1. Short summary
         2. How to use this API
     """

    prompt_template = PromptTemplate(input_variables=["use_case"], template=template)

    # Connected to OpenAI API and using openai package underneath
    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")

    chain = LLMChain(llm=llm, prompt=prompt_template)

    print(chain.run(use_case="Submit Sql queries to my lakehouse."))
