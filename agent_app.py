import os
from datetime import datetime
from typing import List, Any

import ollama
import sys

from akshare_tools import *

# --- 配置 ---
DB_NAME = 'ollama_financial_agent.db'
MODEL_NAME = 'gpt-oss:20b'


# --- 日志配置修改 ---
LOG_DIR = 'debug'
LOG_FILENAME = datetime.now().strftime(f'{LOG_DIR}/%Y%m%d_%H%M%S.log')

if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)
logging.basicConfig(level=logging.NOTSET, handlers=[])
formatter = logging.Formatter(
    '[%(asctime)s] - %(name)s - %(levelname)s - %(message)s'
)

file_handler = logging.FileHandler(LOG_FILENAME, encoding='utf-8')
file_handler.setLevel(logging.DEBUG)  # 文件中记录所有细节
file_handler.setFormatter(formatter)
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.FATAL)
console_handler.setFormatter(logging.Formatter('[%(levelname)s] %(message)s'))

root_logger = logging.getLogger()
root_logger.setLevel(logging.DEBUG)
root_logger.addHandler(file_handler)
root_logger.addHandler(console_handler)

# 设置我们自己的 logger 实例
logger = logging.getLogger(__name__)

logger.info(f"日志系统初始化完成。详细日志已写入：{LOG_FILENAME}")

# 这个系统提示告诉 LLM 它可以调用哪些函数以及它们的 JSON 描述
TOOL_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "get_stock_zh_a_spot_data",
            "description": "获取 A 股的历史行情数据，并将数据存入 SQLite 数据库的固定表中。返回包含最新价格的摘要。",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "description": "股票代码或指数代码，如 '600519' 或 'sh000001'"},
                    "period": {"type": "string", "description": "数据周期 ('daily', 'weekly', 'monthly')"},
                    "start_date": {"type": "string", "description": "开始日期，格式 'YYYYMMDD'"},
                    "end_date": {"type": "string", "description": "结束日期，格式 'YYYYMMDD'"}
                },
                "required": ["symbol"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "visualize_stock_data_trend",
            "description": "从数据库中读取固定表中的数据，计算滑动平均线，生成并显示价格趋势图。用于响应'画图'、'走势图'等请求。注意：此工具需在'get_stock_zh_a_spot_data'调用后使用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "description": "股票代码，仅用于图表标题。"}
                },
                "required": ["symbol"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "query_macro_data",
            "description": "查询中国的年度宏观经济数据，如 CPI 或 GDP。用于响应'最新CPI'、'GDP数据'等请求。数据来自 AkShare。",
            "parameters": {
                "type": "object",
                "properties": {
                    "indicator": {
                        "type": "string",
                        "description": "要查询的宏观经济指标，目前仅支持 'CPI' (年度) 或 'GDP' (年度)。",
                        "enum": ["CPI", "GDP"]
                    }
                },
                "required": ["indicator"]
            }
        }
    }
]

tool_schema_str = json.dumps(TOOL_SCHEMA, indent=2)

SYSTEM_PROMPT_TEMPLATE = """
你是一个专业的金融数据分析助手。你的主要目标是帮助用户获取、处理和可视化股票和指数数据。

---
核心指令：
1. **任务与工具匹配：** 如果用户的请求涉及到获取市场数据（如价格、历史数据、趋势、走势图），你**必须**使用提供的工具。
2. **工具调用：** 如果你决定调用工具，请直接按照 Ollama/GPT-OSS 规范返回结构化的 `tool_calls` 字段。
3. **最终回复：** 在工具调用完成后，基于工具返回的结果进行专业的分析和回复。
4. **简洁性：** 除非必要，避免冗长或不相关的讨论。

你拥有以下工具（JSON Schema）：

{}

---
"""

SYSTEM_PROMPT = SYSTEM_PROMPT_TEMPLATE.format(tool_schema_str)# 拼合SYSTEM_PROMPT_TEMPLATE与TOOL_SCHEMA


def init_db():  # 初始化数据库
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("""
                   CREATE TABLE IF NOT EXISTS conversation
                   (
                       id
                       INTEGER
                       PRIMARY
                       KEY
                       AUTOINCREMENT,
                       role
                       TEXT
                       NOT
                       NULL,
                       content
                       TEXT
                       NOT
                       NULL,
                       timestamp
                       DATETIME
                       DEFAULT
                       CURRENT_TIMESTAMP
                   )
                   """)
    conn.commit()
    conn.close()
    logger.info(f"SQLite 数据库 '{DB_NAME}' 初始化完成.")


def save_message(role, content):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO conversation (role, content) VALUES (?, ?)",
        (role, content)
    )
    conn.commit()
    conn.close()
    logger.info(f"消息已保存: Role='{role}' | Content Length={len(content)}")


def load_context():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT role, content FROM conversation ORDER BY timestamp ASC")
    history_records = cursor.fetchall()
    conn.close()

    messages = []
    # 历史记录中的 assistant 消息可能是 Tool Call 或普通响应，都需要保持
    for role, content in history_records:
        messages.append({'role': role, 'content': content})

    logger.info(f"已加载 {len(messages)} 条历史消息作为上下文.")
    return messages


def clear_screen():
    os.system('cls')


# Tool Calling 解析与执行

def execute_tool_call(tool_calls: List[Any]) -> str:    # 直接接收模型返回的 ToolCall 对象列表，执行工具，并返回 JSON 字符串结果。
    if not tool_calls or not isinstance(tool_calls, list):
        return json.dumps({"error": "工具执行失败: 输入不是有效的 ToolCall 列表。"})

    # 提取第一个工具调用（假设只处理第一个）
    try:
        tool_call = tool_calls[0]

        # 提取工具名称和参数
        tool_name = tool_call.function.name
        tool_args = tool_call.function.arguments  # Dict[str, Any]

    except AttributeError:
        # 处理属性访问错误，如果结构与预期不符
        # 在更换为 gpt-oss:20b 后有很大改善
        return json.dumps({"error": "工具执行失败: 无法从 ToolCall 对象中解析出 function/name/arguments 属性。"})
    except Exception as e:
        return json.dumps({"error": f"工具执行失败: 解析 ToolCall 时发生未知错误: {e}"})

    # 2. 查找并执行工具
    try:
        if tool_name not in AVAILABLE_TOOLS:
            return json.dumps({"error": f"找不到工具: {tool_name}"})

        tool_function = AVAILABLE_TOOLS[tool_name]

        # 执行工具
        tool_result = tool_function(**tool_args)

        return tool_result  # tool_result 已经是 JSON 字符串，保持返回类型一致

    except Exception as e:
        # 记录详细错误信息
        import traceback
        traceback.print_exc()
        return json.dumps({"error": f"工具函数 '{tool_name}' 执行失败: {e}"})


def chat_with_context(user_input): # 主对话逻辑
    # 1. 准备消息列表 (包含系统提示和历史记录)
    context_messages = load_context()
    context_messages.insert(0, {'role': 'system', 'content': SYSTEM_PROMPT})
    context_messages.append({'role': 'user', 'content': user_input})

    # 将最新的用户输入保存到数据库
    save_message('user', user_input)

    # 2. 初始 Ollama 调用 (可能会返回 Tool Call)
    print(f"\n👤 你: {user_input}\n")
    print("🤖 Agent 处理中...")


    # LLM 响应的第一部分
    response = ollama.chat(
        model=MODEL_NAME,
        messages=context_messages,
        stream=False,
    )

    message = response['message']

    # 核心判断逻辑
    # 检查是否有 tool_calls 字段，且它是一个非空的列表
    if 'tool_calls' in message and isinstance(message['tool_calls'], list) and message['tool_calls']:

        tool_call_list = message['tool_calls']

        # 提取第一个工具调用
        first_tool_call = tool_call_list[0]
        tool_function = first_tool_call['function']

        tool_name = tool_function['name']
        tool_args = tool_function['arguments']

        # 构建 JSON 字符串用于数据库和 LLM 的第二次调用
        tool_call_data = {"tool": tool_name, "arguments": tool_args}
        tool_call_json = json.dumps(tool_call_data)

        # 3. 工具执行逻辑
        print(f"Agent: **已识别到工具调用**，正在执行...")
        save_message('assistant', tool_call_json)
        logger.info(f"LLM 识别为工具调用，正在执行：{tool_call_json[:100]}...")

        tool_output = execute_tool_call(message.tool_calls)

        # 4. 第二次 Ollama 调用 (带着工具结果)
        logger.info("进行第二次 LLM 调用 (带工具结果) 以获取最终回复...")

        # 4a. 准备第二次调用的消息列表 (保持不变)
        second_call_messages = [{'role': 'system', 'content': SYSTEM_PROMPT}] + context_messages.copy()
        second_call_messages.append({'role': 'user', 'content': user_input})
        second_call_messages.append({'role': 'assistant', 'content': tool_call_json})
        second_call_messages.append({'role': 'tool', 'content': tool_output})

        # 4b. 第二次流式调用
        final_response = ollama.chat(
            model=MODEL_NAME,
            messages=second_call_messages,
            stream=False
        )

        # 4c. 提取最终响应
        final_answer = final_response.message.content.strip()

        print("\n   Agent 最终回复: ")
        print(final_answer)  # 直接打印完整回复

        print("\n" + "-" * 30 + "\n")
    else:

        # 提取 content 字段作为最终回复
        full_response_content = message.get('content', '').strip()

        if not full_response_content:
            # 如果模型没有返回 content，但也没有 tool_calls，可能是闲聊模式的思考过程
            full_response_content = "我没有找到可以执行的工具，请明确您的问题。"

        # 3. 直接回复
        print(" Agent 最终回复: ")
        print(full_response_content)
        print("\n" + "-" * 30 + "\n")

        # 4. 保存第一个响应
        save_message('assistant', full_response_content)

# 主循环
def interactive_chat():
    init_db()

    while True:
        try:
            user_input = input("请输入你的金融问题 (输入 'exit' 退出, 'clear' 清空历史): \n> ")
        except EOFError:
            break

        if user_input.lower() == 'exit':
            break

        if user_input.lower() == 'clear':
            conn = sqlite3.connect(DB_NAME)
            conn.execute("DELETE FROM conversation")
            conn.commit()
            conn.close()
            logger.info("对话历史已清空！")
            continue

        if user_input.strip():
            chat_with_context(user_input)

    logger.info("金融 AI Agent 已退出。")


if __name__ == '__main__':
    clear_screen()
    interactive_chat()
    # 加载并显示历史概要
    history = load_context()
    if history:
        print(f"--- 历史记录 ({len(history)} 条消息) ---")
        last_user_msg = next((m['content'] for m in reversed(history) if m['role'] == 'user'), "无")
        print(f"最近用户: {last_user_msg[:50]}...")
        print("-" * 30)
    else:
        print("--- 开始新对话 ---")