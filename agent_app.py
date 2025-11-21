# 主程序

import os
from datetime import datetime
from typing import List, Any, Dict

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
你是一个专业的金融数据分析智能体 (Agent)。你的核心职责是准确调用工具获取数据并进行分析。

### 🚨 核心原则 (Critical Rules)
1. **禁止猜测数据：** 任何涉及股票价格、历史行情、财务数据的请求，**必须**通过调用工具获取，严禁根据训练数据编造。
2. **直接函数调用：** - 严禁创造不存在的函数（如 `assistant`, `call_tool`, `api_caller` 等）。
   - **直接使用**工具列表中定义的函数名（如 `get_stock_zh_a_spot_data`）。
3. **参数精确匹配：** 严格遵守工具定义中的参数名和日期格式（通常为 "YYYY-MM-DD"）。

### 📝 调用范例 (Few-Shot Examples)

**正确示范 (Correct):**
用户: "帮我查一下茅台(600519)最近的行情"
模型行为: 调用函数 `get_stock_zh_a_spot_data`
参数: {{"symbol": "600519"}}

**错误示范 (Wrong - 不要这样做):**
❌ 错误 1 (嵌套调用): {{"function": {{"name": "assistant", "arguments": {{"tool": "get_stock_zh_a_spot_data", ...}}}}}}
❌ 错误 2 (错误的函数名): 调用 `search_stock` (如果工具表中没有这个函数)

### 🛠️ 可用工具定义 (Tool Definitions)
以下是你唯一允许使用的工具：

{}

---
请根据用户输入，一步步思考，如果需要数据，立即生成 Tool Call。
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
# 反射
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


def chat_with_context(user_input):
    """
    支持连续工具调用的主对话逻辑
    """
    # 1. 准备初始上下文
    # 注意：load_context 返回的通常是历史记录列表
    history = load_context()

    # 构建当前会话的消息列表
    messages = []
    messages.append({'role': 'system', 'content': SYSTEM_PROMPT})
    messages.extend(history) # 添加历史记录
    messages.append({'role': 'user', 'content': user_input})

    # 保存用户输入到数据库
    save_message('user', user_input)

    print(f"\n👤 你: {user_input}\n")

    # 设置最大循环次数，防止死循环 (例如模型不断报错不断重试)
    MAX_ITERATIONS = 5
    iteration = 0

    while iteration < MAX_ITERATIONS:
        print("🤖 Agent 思考中...")

        # 2. 调用 Ollama
        response = ollama.chat(
            model=MODEL_NAME,
            messages=messages,
            stream=False,
        )

        response_message = response['message']

        # 将模型的回复添加到消息历史中 (无论是文本还是工具调用，都必须加进去，否则模型会忘记它刚才做了什么)
        messages.append(response_message)

        # 3. 检查是否有工具调用
        if response_message.get('tool_calls'):
            # 获取原始对象列表
            raw_tool_calls = response_message['tool_calls']

            # ==================== 核心修复开始 ====================
            # 将 ToolCall 对象列表转换为普通的字典列表 (Dict List)
            # 这样既可以通过 json.dumps 保存，也可以通过 ['key'] 下标访问
            tool_calls_serializable = []

            for tool in raw_tool_calls:
                if isinstance(tool, dict):
                    # 如果已经是字典，直接添加
                    tool_calls_serializable.append(tool)
                elif hasattr(tool, 'model_dump'):
                    # Ollama (Pydantic v2) 通常有 model_dump 方法
                    tool_calls_serializable.append(tool.model_dump())
                elif hasattr(tool, 'dict'):
                    # 旧版本 Pydantic 可能用 .dict()
                    tool_calls_serializable.append(tool.dict())
                else:
                    # 如果以上都没有，手动提取属性
                    tool_calls_serializable.append({
                        'function': {
                            'name': tool.function.name,
                            'arguments': tool.function.arguments
                        },
                        'type': 'function'
                    })
            # ==================== 核心修复结束 ====================

            # 1. 保存到数据库 (现在传入的是字典列表，JSON 序列化不会报错了)
            save_message('assistant', json.dumps(tool_calls_serializable, ensure_ascii=False))

            print(f"Agent: **需要调用 {len(tool_calls_serializable)} 个工具**，正在执行...")

            # 2. 遍历执行 (注意：这里遍历我们要用转换后的 tool_calls_serializable)
            for tool in tool_calls_serializable:
                # 因为我们已经转成了字典，所以这里可以用 ['key'] 访问，不会报错
                function_name = tool['function']['name']
                function_args = tool['function']['arguments']

                logger.info(f"正在执行工具: {function_name} 参数: {function_args}")
                try:
                    tool_output = execute_single_tool(function_name,function_args)

                except Exception as e:
                    tool_output = f"Tool execution error: {str(e)}"

                # 4. 将工具执行结果作为 'tool' 角色消息添加回去
                tool_message = {
                    'role': 'tool',
                    'content': str(tool_output),
                    # 某些 API 可能需要 tool_call_id，Ollama 目前主要依赖顺序，但加上更稳妥
                    # 'name': function_name
                }
                messages.append(tool_message)

                # 同时也保存工具结果到数据库，以便未来上下文使用
                save_message('tool', str(tool_output))

            # 循环继续：带着工具结果回到开头，再次调用 ollama.chat
            iteration += 1
            logger.info(f"工具执行完毕，进入第 {iteration} 轮思考...")

        else:
            # 5. 没有工具调用 -> 最终回复
            final_content = response_message.get('content', '').strip()

            if not final_content:
                final_content = "任务已完成，但我没有更多内容要补充。"

            print("\n   Agent 最终回复: ")
            print(final_content)
            print("\n" + "-" * 30 + "\n")

            # 保存最终回复
            save_message('assistant', final_content)

            # 任务结束，跳出循环
            return

    # 如果循环次数用尽
    print("⚠️ 达到最大对话轮数限制，停止执行。")
'''
def chat_with_context(user_input): # 主对话逻辑
    # 1. 准备消息列表 (包含系统提示和历史记录)
    context_messages = load_context()# 初始化上下文
    context_messages.insert(0, {'role': 'system', 'content': SYSTEM_PROMPT})
    context_messages.append({'role': 'user', 'content': user_input})

    save_message('user', user_input) # 将最新的用户输入保存到数据库

    # 2. 初始 Ollama 调用
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

        tool_output = execute_single_tool(message.tool_calls)

        # 4. 第二次 Ollama 调用 (带着工具结果)
        logger.info("进行第二次 LLM 调用 (带工具结果) 以获取最终回复...")

        # 4a. 准备第二次调用的消息列表 (保持不变)
        second_call_messages = [{'role': 'system', 'content': SYSTEM_PROMPT}] + context_messages.copy()
        second_call_messages.append({'role': 'user', 'content': user_input})
        second_call_messages.append({'role': 'assistant', 'content': tool_call_json})
        second_call_messages.append({'role': 'tool', 'content': tool_output})

        # 4b. 第二次 Ollama 调用
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
'''
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