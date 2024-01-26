from dotenv import load_dotenv, find_dotenv
import semantic_kernel as sk
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion
import os
import gradio as gr

# 自定义 system message
system_message = """
你是一个调皮的机器人，幽默地回答任何问题
"""
# 加载 .env 到环境变量
_ = load_dotenv(find_dotenv())

# 创建 semantic kernel
kernel = sk.Kernel()

# 配置 OpenAI 服务
api_key = os.getenv('OPENAI_API_KEY')
endpoint = os.getenv('OPENAI_API_BASE')
model = OpenAIChatCompletion(
    "gpt-3.5-turbo", api_key, endpoint=endpoint)

# 把对话服务加入 kernel
kernel.add_chat_service("chat-gpt", model)


async def chat(prompt, history):

    # 创建 prompt template
    prompt_config = sk.PromptTemplateConfig.from_completion_parameters(
        max_tokens=2000, temperature=0.7
    )
    prompt_template = sk.ChatPromptTemplate(
        "{{$user_input}}", kernel.prompt_template_engine, prompt_config
    )
    prompt_template.add_system_message(system_message)
    for human_message, ai_message in history:
        prompt_template.add_user_message(human_message)
        prompt_template.add_assistant_message(ai_message)

    # 创建 semantic function
    function_config = sk.SemanticFunctionConfig(prompt_config, prompt_template)
    chat_function = kernel.register_semantic_function(
        "ChatBot", "Chat", function_config)

    # 定义上下文变量
    context_vars = sk.ContextVariables()
    context_vars["user_input"] = prompt

    # 流式调用 LLM
    answer = []
    async for message in kernel.run_stream_async(chat_function, input_vars=context_vars):
        if (message.strip() == "assistant:"):
            continue
        answer.append(message)
        yield "".join(answer).strip()

# 初始化 gradio
demo = gr.ChatInterface(chat)

# 启动 gradio
demo.queue().launch()
