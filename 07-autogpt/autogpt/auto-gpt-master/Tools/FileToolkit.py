from langchain.agents.agent_toolkits import FileManagementToolkit

file_toolkit = FileManagementToolkit(
    root_dir="./temp"
)  # If you don't provide a root_dir, operations will default to the current working directory