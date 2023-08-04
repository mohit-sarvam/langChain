from langchain import PromptTemplate

import langchain_visualizer
from langchain.chains import LLMMathChain

from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.agents import initialize_agent, Tool, AgentType

from third_parties.databricks import get_documentation, get_full_documentation

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

    prompt = PromptTemplate(input_variables=["query"], template="{query}")
    llm_chain = LLMChain(llm=llm, prompt=prompt)
    llm_math = LLMMathChain(llm=llm)

    tools_for_agent = [
        Tool(
            name="Docs",
            func=get_full_documentation,
            description="useful for databricks documentation",
        ),
        Tool(
            name="Calculator",
            func=llm_math.run,
            description="Useful for when you need to answer questions about math.",
        ),
        Tool(
            name="Language Model",
            func=llm_chain.run,
            description="use this tool for general purpose queries and logic",
        ),
    ]

    agent = initialize_agent(
        tools=tools_for_agent,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
    )

    async def async_run_agent():
        return agent.run(
            prompt_template.format_prompt(
                use_case="Submit Sql queries to my lakehouse."
            )
        )

    langchain_visualizer.visualize(async_run_agent)
