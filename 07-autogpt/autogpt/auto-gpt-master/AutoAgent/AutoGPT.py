import langchain
from typing import List, Optional
from colorama import init
from langchain import OpenAI
from langchain.chat_models import ChatOpenAI

from langchain.chat_models.base import BaseChatModel
from langchain.llms import BaseLLM
from langchain.memory import ConversationTokenBufferMemory, VectorStoreRetrieverMemory, ChatMessageHistory, \
    ConversationBufferWindowMemory
from langchain.output_parsers import PydanticOutputParser, OutputFixingParser
from langchain.tools.base import BaseTool
from langchain.chains.llm import LLMChain
from langchain.vectorstores.base import VectorStoreRetriever
from pydantic import ValidationError
from langchain.memory import ConversationSummaryMemory

from Utils.ThoughtAndAction import ThoughtAndAction, Action, Thought
from Utils.PromptTemplateBuilder import PromptTemplateBuilder

def format_thought(thought: Thought)->str:
    def format_plans(plans: List[str]):
        ans = ""
        for plan in plans:
            ans += f" - {plan}\n"
        return ans.strip()

    ans = (
        "\n"
        f"思考: {thought.text}\n"
        f"推理: {thought.reasoning}\n"
        f"计划: {format_plans(thought.plan)}\n"
        f"反思: {thought.criticism}\n"
        f"输出: {thought.speak}\n"
        "\n"
    )
    return ans


def format_action(action: Action):
    ans = f"{action.name}("
    if action.args is None or len(action.args) == 0:
        ans += ")"
        return ans
    for k,v in action.args.items():
        ans += f"{k}={v},"
    ans = ans[:-1]+")"
    return ans


class AutoGPT:
    """AutoGPT：基于Langchain实现"""

    def __init__(
            self,
            llm: BaseLLM | BaseChatModel,
            prompts_path: str,
            tools: List[BaseTool],
            agent_name: Optional[str] = "瓜瓜",
            agent_role: Optional[str] = "强大的AI助手，可以使用工具与指令自动化解决问题",
            max_thought_steps: Optional[int] = 10,
            memery_retriever: Optional[VectorStoreRetriever] = None,
    ):
        self.llm = llm
        self.prompts_path = prompts_path
        self.tools = tools
        self.agent_name = agent_name
        self.agent_role = agent_role
        self.max_thought_steps = max_thought_steps
        self.memery_retriever = memery_retriever

        self.output_parser = PydanticOutputParser(pydantic_object=ThoughtAndAction)

        self.step_prompt = PromptTemplateBuilder(self.prompts_path, "step_instruction.templ").build().format()
        self.force_rethink_prompt = PromptTemplateBuilder(self.prompts_path, "force_rethink.templ").build().format()


    def _find_tool(self, tool_name: str) -> Optional[BaseTool]:
        for tool in self.tools:
            if tool.name == tool_name:
                return tool
        return None

    def _is_repeated(self, last_action: Action, action: Action) -> bool:
        """判断两个Action是否重复"""
        if last_action is None:
            return False
        if last_action.name != action.name:
            return False
        if last_action.args is None and action.args is None:
            return True
        if last_action.args is None or action.args is None:
            return False
        for k,v in last_action.args.items():
            if k not in action.args:
                return False
            if action.args[k] != v:
                return False
        return True

    def _step(self,
              chain,
              task_description,
              short_term_memory,
              long_term_memory,
              force_rethink=False,
        ) -> ThoughtAndAction:
        """执行一步思考"""
        current_response = chain.run(
            short_term_memory=short_term_memory.load_memory_variables({})["history"],
            long_term_memory=long_term_memory.load_memory_variables(
                {"prompt": task_description}
            )["history"] if long_term_memory is not None else "",
            step_instruction=self.step_prompt if not force_rethink else self.force_rethink_prompt,
        )
        # OutputFixingParser： 如果输出格式不正确，尝试修复
        robust_parser = OutputFixingParser.from_llm(parser=self.output_parser, llm=self.llm)
        thought_and_action = robust_parser.parse(current_response)
        return thought_and_action

    def _final_step(self, short_term_memory, task_description) -> str:
        """最后一步, 生成最终的输出"""
        finish_prompt = PromptTemplateBuilder(
            self.prompts_path,
            "finish_instruction.templ"
        ).build().partial(
            ai_name=self.agent_name,
            ai_role=self.agent_role,
            task_description=task_description,
            short_term_memory=short_term_memory.load_memory_variables({})["history"]
        )
        chain = LLMChain(llm=self.llm, prompt=finish_prompt)
        response = chain.run({})
        return response

    def run(self, task_description, verbose=False) -> str:
        thought_step_count = 0 # 思考步数

        # 初始化模板
        prompt_template = PromptTemplateBuilder(self.prompts_path).build(
            tools=self.tools,
            output_parser=self.output_parser,
        ).partial(
            ai_name=self.agent_name,
            ai_role=self.agent_role,
            task_description=task_description,
        )
        # 初始化LLM链
        chain = LLMChain(llm=self.llm, prompt=prompt_template)

        short_term_memory = ConversationBufferWindowMemory(
            ai_prefix="Reason",
            human_prefix="Act",
            k=self.max_thought_steps,
        )

        # 用于生成思考过程总结（存入长时记忆中）
        summary_memory = ConversationSummaryMemory(
            llm=OpenAI(temperature=0),
            buffer="问题: "+task_description+"\n",
            ai_prefix="Reason",
            human_prefix="Act",
        )

        # 如果有长时记忆，加载长时记忆
        if self.memery_retriever is not None:
            long_term_memory = VectorStoreRetrieverMemory(
                retriever=self.memery_retriever
            )
        else:
            long_term_memory = None

        reply = ""
        finish_turn = False
        last_action = None

        while thought_step_count < self.max_thought_steps:

            # 如果是最后一步，执行最后一步
            if finish_turn:
                reply = self._final_step(short_term_memory, task_description)
                break

            # 执行一步思考
            thought_and_action = self._step(
                chain=chain,
                task_description=task_description,
                short_term_memory=short_term_memory,
                long_term_memory=long_term_memory,
            )

            action = thought_and_action.action

            # 如果是重复的Action，强制重新思考
            if self._is_repeated(last_action, action):
                thought_and_action = self._step(
                    chain=chain,
                    task_description=task_description,
                    short_term_memory=short_term_memory,
                    long_term_memory=long_term_memory,
                    force_rethink=True,
                )
                action = thought_and_action.action

            last_action = action

            if verbose:
                print(format_thought(thought_and_action.thought))

            if thought_and_action.is_finish():
                finish_turn = True
                continue

            # 查找工具
            tool = self._find_tool(action.name)
            if tool is None:
                result = (
                    f"Error: 找不到工具或指令 '{action.name}'. "
                    f"请从提供的工具/指令列表中选择，请确保按对顶格式输出。"
                )
            else:
                try:
                    # 执行工具
                    observation = tool.run(action.args)
                except ValidationError as e:
                    # 工具的入参异常
                    observation = (
                        f"Validation Error in args: {str(e)}, args: {action.args}"
                    )
                except Exception as e:
                    # 工具执行异常
                    observation = (
                        f"Error: {str(e)}, {type(e).__name__}, args: {action.args}"
                    )
                result = (
                    f"执行: {format_action(action)}\n" 
                    f"返回结果: {observation}"
                )
            if verbose:
                print(result)

            # 保存到短时记忆
            short_term_memory.save_context(
                {"input": format_thought(thought_and_action.thought)},
                {"output": result}
            )

            if long_term_memory is not None:
                # 保存到总结记忆
                summary_memory.save_context(
                    {"input": format_thought(thought_and_action.thought)},
                    {"output": format_action(action)}
                )

            thought_step_count += 1
            reply = thought_and_action.thought.speak

        if long_term_memory is not None:
            # 保存到长时记忆
            long_term_memory.save_context(
                {"input": task_description},
                {"output": summary_memory.load_memory_variables({})["history"]}
            )

        return reply

