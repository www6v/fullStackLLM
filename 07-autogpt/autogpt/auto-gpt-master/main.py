from dotenv import load_dotenv
load_dotenv("my_api_keys.env") # 从环境变量中加载 API keys，必须在所有 import 之前

from AutoAgent.AutoGPT import AutoGPT
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.schema import Document
from Tools import *


tools = [
    search_tool,
    map_tool,
    mocked_location_tool,
    calculator_tool,
    calendar_tool,
    weather_tool,
    webpage_tool,
] + file_toolkit.get_tools()


def main():
    llm = ChatOpenAI(model="gpt-4", temperature=0)
    prompts_path = "prompts"
    db = Chroma.from_documents([Document(page_content="")], OpenAIEmbeddings(model="text-embedding-ada-002"))
    retriever = db.as_retriever(search_kwargs=dict(k=1))
    agent = AutoGPT(
        llm=llm,
        prompts_path=prompts_path,
        tools=tools,
        max_thought_steps=10,
        memery_retriever=retriever
    )

    while True:
        task = input("有什么可以帮您:\n>>>")
        if task.strip().lower() == "quit":
            break
        reply = agent.run(task,verbose=True)
        print(f"{agent.agent_name}: {reply}\n")


if __name__ == "__main__":
    main()
