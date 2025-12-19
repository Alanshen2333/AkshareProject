import logging
import sqlite3
import json
import uuid
from abc import ABC, abstractmethod
from typing import Dict, Any

import ollama

from agent_app import save_message, load_context, SYSTEM_PROMPT
from akshare_tools import AVAILABLE_TOOLS
from utils.context_store import ContextStore

MODEL_NAME = 'gpt-oss:20b'
LOG_DIR = 'debug'
logging.basicConfig(
        level=logging.INFO,
        format=(
            "%(asctime)s | %(levelname)s | "
            "task=%(task_id)s | agent=%(agent)s | "
            "%(message)s"
        ),
        datefmt="%Y-%m-%d %H:%M:%S",)


class ChatHistoryDB:
    def __init__(self, db_name="agent_memory.db"):
        self.conn = sqlite3.connect(db_name)
        self.cursor = self.conn.cursor()
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS chat_logs 
            (id INTEGER PRIMARY KEY AUTOINCREMENT, 
             agent_name TEXT, 
             input_text TEXT, 
             output_text TEXT, 
             timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)
        ''')
        self.conn.commit()

    def save_log(self, agent_name, input_data, output_data):
        self.cursor.execute(
            "INSERT INTO chat_logs (agent_name, input_text, output_text) VALUES (?, ?, ?)",
            (agent_name, str(input_data), str(output_data))
        )
        self.conn.commit()


class AgentContext:
    def __init__(self, user_input: str):
        self.user_input = user_input
        self.messages = []
        self.iteration = 0
        self.finished = False

class Agent(ABC):
    def __init__(self, name: str, store: ContextStore):
        self.name = name
        self.store = store
        self.logger = None

    def _bind_logger(self, task_id: str):
        base_logger = logging.getLogger(self.name)
        self.logger = logging.LoggerAdapter(
            base_logger,
            {
                "task_id": task_id,
                "agent": self.name,
            }
        )

    @abstractmethod
    def run(self, ctx: dict) -> None:
        pass

    def persist(self, ctx: dict):
        self.store.save_context(ctx["task_id"], ctx, self.name)
        self.logger.info("context persisted to sqlite")

class DataFetchAgent(Agent):
    def run(self, ctx: dict) -> None:
        print("📡 DataFetchAgent")

        ctx["raw_data"] = {
            "source": "demo",
            "values": [1, 2, 3, 4, 5]
        }

        self.persist(ctx)

def execute_single_tool(tool_name: str, tool_args: Dict[str, Any]) -> str:
    """
    执行单个工具调用。
    接收函数名和参数字典，返回 JSON 字符串结果。
    """

    # 1. 查找工具
    if tool_name not in AVAILABLE_TOOLS:
        error_msg = f"执行失败: 找不到工具 '{tool_name}'"
        print(f"❌ {error_msg}")
        return json.dumps({"error": error_msg}, ensure_ascii=False)

    tool_function = AVAILABLE_TOOLS[tool_name]

    try:
        # 2. 执行工具
        # 使用 ** 解包字典参数传入函数
        print(f"⚙️ 正在调用: {tool_name}({tool_args})")
        result = tool_function(**tool_args)

        # 3. 统一返回值格式
        # LLM 需要接收 String 类型的 content。
        # 如果工具返回的是字典、列表等对象，必须转为 JSON 字符串。
        if isinstance(result, (dict, list, int, float, bool)):
            return json.dumps(result, ensure_ascii=False)

        # 如果已经是字符串（例如 JSON 字符串），直接返回
        return str(result)

    except TypeError as e:
        # 捕捉参数错误（例如模型幻觉生成了不存在的参数）
        error_msg = f"参数错误: 工具 '{tool_name}' 不接受提供的参数: {e}"
        print(f"❌ {error_msg}")
        return json.dumps({"error": error_msg}, ensure_ascii=False)

    except Exception as e:
        # 捕捉工具内部运行时的其他异常# 打印堆栈信息方便调试
        error_msg = f"运行时错误: 工具 '{tool_name}' 执行异常: {str(e)}"
        return json.dumps({"error": error_msg}, ensure_ascii=False)


TOOL_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "get_stock_zh_a_spot_data",
            "description": "获取 A 股的历史行情数据，并将数据存入 SQLite 数据库的固定表中。返回包含最新价格的摘要。",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string"},
                    "period": {"type": "string"},
                    "start_date": {"type": "string"},
                    "end_date": {"type": "string"}
                },
                "required": ["symbol"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "visualize_stock_data_trend",
            "description": "从数据库读取数据并生成走势图（需在数据获取后调用）",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string"}
                },
                "required": ["symbol"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "query_macro_data",
            "description": "查询中国年度宏观经济数据（CPI / GDP）",
            "parameters": {
                "type": "object",
                "properties": {
                    "indicator": {"type": "string", "enum": ["CPI", "GDP"]}
                },
                "required": ["indicator"]
            }
        }
    }
]

TOOL_SCHEMA_STR = json.dumps(TOOL_SCHEMA, indent=2, ensure_ascii=False)
DATA_FETCH_PROMPT = f"""
你是【数据获取 Agent】。

职责：
- 判断是否需要真实金融/宏观数据
- 必须通过工具获取
- 数据会自动存入 SQLite

🚨 规则：
1. 严禁猜测任何数值
2. 涉及行情 / CPI / GDP → 必须调用工具
3. 只允许使用：
   - get_stock_zh_a_spot_data
   - query_macro_data
4. 不得进行分析或总结

可用工具：
{TOOL_SCHEMA_STR}
"""

DATA_PROCESS_PROMPT = """
你是【数据处理 Agent】。

职责：
- 基于已获取并存入 SQLite 的数据进行分析
- 在用户请求“画图 / 走势图”时生成图表

🚨 规则：
1. 禁止重新获取数据
2. 只允许使用 visualize_stock_data_trend
3. 若缺少前置数据，必须说明

不写最终报告
"""

REPORT_PROMPT = """
你是【报告撰写 Agent】。

职责：
- 将已有分析结果整理为最终报告

🚨 规则：
1. 严禁调用任何工具
2. 不补充任何未给出的数据
3. 不提 Agent / 工具 / 数据库

直接输出最终报告
"""

class DataFetchAgent(Agent):
    def run(self, ctx: dict):
        self.bind_logger(ctx["task_id"])
        self.logger.info("start")

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": ctx["user_input"]},
        ]

        resp = ollama.chat(
            model=ctx["model"],
            messages=messages,
            tools=TOOL_SCHEMA,
        )

        msg = resp["message"]
        ctx["messages"].append(msg)

        if msg.get("tool_calls"):
            for tool in msg["tool_calls"]:
                result = execute_single_tool(
                    tool["function"]["name"],
                    tool["function"]["arguments"],
                )
                ctx["messages"].append({
                    "role": "tool",
                    "content": str(result),
                })

        self.store.save(ctx["task_id"], ctx)
        self.logger.info("finished")


class DataProcessAgent(Agent):
    def run(self, ctx: dict):
        self.bind_logger(ctx["task_id"])
        self.logger.info("start")

        messages = [
            {"role": "system", "content": self.system_prompt},
            *ctx["messages"],
        ]

        resp = ollama.chat(
            model=ctx["model"],
            messages=messages,
            tools=TOOL_SCHEMA,
        )

        msg = resp["message"]
        ctx["messages"].append(msg)

        if msg.get("tool_calls"):
            for tool in msg["tool_calls"]:
                result = execute_single_tool(
                    tool["function"]["name"],
                    tool["function"]["arguments"],
                )
                ctx["messages"].append({
                    "role": "tool",
                    "content": str(result),
                })

        self.store.save(ctx["task_id"], ctx)
        self.logger.info("finished")


class ReportAgent(Agent):
    def run(self, ctx: dict):
        self.bind_logger(ctx["task_id"])
        self.logger.info("start")

        messages = [
            {"role": "system", "content": self.system_prompt},
            *ctx["messages"],
        ]

        resp = ollama.chat(
            model=ctx["model"],
            messages=messages,
            stream=False,
        )

        ctx["final_report"] = resp["message"]["content"]
        self.store.save(ctx["task_id"], ctx)

        self.logger.info("finished")


def run(user_input: str):

    store = ContextStore()
    task_id = str(uuid.uuid4())

    ctx = {
        "task_id": task_id,
        "user_input": user_input,
        "model": "qwen2.5:14b",
        "messages": [],
        "final_report": None,
    }

    store.save(task_id, ctx)

    agents = [
        DataFetchAgent("DataFetchAgent", DATA_FETCH_PROMPT, store),
        DataProcessAgent("DataProcessAgent", DATA_PROCESS_PROMPT, store),
        ReportAgent("ReportAgent", REPORT_PROMPT, store),
    ]

    for agent in agents:
        agent.run(ctx)

    print("\n===== 最终报告 =====\n")
    print(ctx["final_report"])