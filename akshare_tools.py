# 工具模块

import sqlite3

import akshare as ak
import pandas as pd
import json
import logging
import seaborn as sns
import matplotlib.pyplot as plt

from typing import Literal

logger = logging.getLogger(__name__)
DB_NAME = 'ollama_financial_agent.db'
TABLE_NAME = 'current_stock_data'
MA_PERIOD = 5 # 数据周期

plt.rcParams['font.sans-serif'] = ['SimHei'] # 指定默认字体为黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号 '-' 显示为方块的问题

def get_db_connection():
    """获取数据库连接"""
    conn = sqlite3.connect(DB_NAME)
    return conn

def get_stock_zh_a_spot_data(symbol: str = "sh000001", period: str = "daily", start_date: str = "20240101",
                             end_date: str = "20241231") -> str:
    """
    获取 A 股的历史行情数据。

    参数:
        symbol (str): 股票代码或指数代码 (例如: '600036' 招商银行, 'sh000001' 上证指数)。
        period (str): 数据周期 ('daily', 'weekly', 'monthly')。
        start_date (str): 开始日期，格式 'YYYYMMDD' (例如 '20240101')。
        end_date (str): 结束日期，格式 'YYYYMMDD' (例如 '20241231')。

    返回:
        str: 包含数据摘要的 JSON 字符串。
    """
    logger.info(f"正在调用 AkShare 获取数据: {symbol}, 周期: {period}")
    try:
        # 1. 获取数据 (保持不变)
        if symbol.startswith(('sh', 'sz')):
            df = ak.index_zh_a_hist(symbol=symbol, period=period, start_date=start_date, end_date=end_date,)
        else:
            df = ak.stock_zh_a_hist(symbol=symbol, period=period, start_date=start_date, end_date=end_date, adjust="qfq")

        if df.empty:
            return json.dumps({"error": f"未找到 {symbol} 在 {start_date} 到 {end_date} 期间的数据。"})

        # 2. 清洗和准备数据 (保持不变)
        df['日期'] = df['日期'].astype(str)

        # 3. 核心操作：存入固定的 TABLE_NAME
        conn = get_db_connection()
        # 存入固定的表，替换现有数据
        df.to_sql(TABLE_NAME, conn, if_exists='replace', index=False)
        conn.close()

        # 4. 返回摘要（供 LLM 回答问题）
        summary = {
            "symbol": symbol,
            "data_points": len(df),
            "start_date": df['日期'].iloc[0],
            "end_date": df['日期'].iloc[-1],
            "latest_close": df['收盘'].iloc[-1],
            "max_high": df['最高'].max(),
        }

        logger.info(f"数据获取成功并存入固定数据库表: {TABLE_NAME}")
        return json.dumps(summary, ensure_ascii=False)

    except Exception as e:
        logger.error(f"AkShare 或数据库操作失败: {e}")
        return json.dumps({"error": f"AkShare/DB 调用失败: {e}"})


# 数据可视化工具

def visualize_stock_data_trend(symbol: str) -> str:
    """
    从固定的 SQLite 表中读取数据，计算滑动平均线，生成并显示图表。

    参数:
        symbol (str): 股票代码，用于图表标题（不再需要 data_table_name）。

    返回:
        str: 包含成功或失败消息的 JSON 字符串。
    """
    logger.info(f"正在从固定表 {TABLE_NAME} 读取数据，准备生成趋势图。")

    try:
        # 从 SQLite 读取数据
        conn = get_db_connection()
        # 固定查询 TABLE_NAME
        df = pd.read_sql(f"SELECT * FROM {TABLE_NAME}", conn)
        conn.close()

        if df.empty:
            return json.dumps({"error": f"固定表 {TABLE_NAME} 中数据为空，请先调用数据获取工具。"})

        # 2. 数据处理与绘图逻辑 (保持不变)
        df['日期'] = pd.to_datetime(df['日期'])
        df.set_index('日期', inplace=True)
        df['收盘'] = pd.to_numeric(df['收盘'], errors='coerce')
        df['开盘'] = pd.to_numeric(df['开盘'], errors='coerce')

        df[f'MA{MA_PERIOD}'] = df['收盘'].rolling(window=MA_PERIOD).mean()

        sns.set_theme(style="whitegrid", font='SimHei', font_scale=1.1)

        plt.figure(figsize=(14, 8))  # 增大图表尺寸

        # 绘制收盘价（突出显示）
        plt.plot(df.index, df['收盘'],
                 label='收盘价',
                 color='#1f77b4',  # 使用更专业的蓝色
                 linewidth=2.5,
                 alpha=0.8,
                 zorder=3)  # 设置层级，让其在前面

        # 绘制开盘价（作为背景参考，线条更细、透明度更高）
        plt.plot(df.index, df['开盘'],
                 label='开盘价',
                 color='#ff7f0e',  # 使用橙色作为对比色
                 linewidth=1.0,
                 linestyle=':',  # 使用虚线
                 alpha=0.6)

        # 绘制滑动平均线（使用红色或专业色，强调趋势）
        ma_label = f'{MA_PERIOD}日均线'
        plt.plot(df.index, df[ma_label],
                 label=ma_label,
                 color='#d62728',  # 使用醒目的红色
                 linewidth=2.5,
                 zorder=4)

        # 标题和标签优化
        plt.title(f'{symbol} 股票价格与 {MA_PERIOD} 日均线趋势分析', fontsize=18, fontweight='bold')
        plt.xlabel('日期', fontsize=14)
        plt.ylabel('价格 (元)', fontsize=14)

        # 增加图例美观度
        plt.legend(loc='upper left', frameon=True, shadow=True, fancybox=True)

        # 调整轴刻度，使其更易读
        plt.tick_params(axis='x', rotation=45, labelsize=10)
        plt.tick_params(axis='y', labelsize=10)

        plt.grid(True, linestyle='-', alpha=0.5)  # 使用实线网格，增强可读性
        plt.tight_layout()

        # 5. 直接显示图表
        plt.show()

        logger.info(f"图表已生成并显示在新的窗口中。")

        return json.dumps({
            "success": True,
            "message": f"成功生成并显示了 {symbol} 股票的趋势图（含{MA_PERIOD}日均线）。请检查新弹出的窗口。"
        })

    except Exception as e:
        logger.error(f"生成或显示图表时发生错误: {e}")
        return json.dumps({"error": f"生成可视化图表失败: {e}。"})


def query_macro_data(indicator: Literal['CPI', 'GDP']) -> str:
    """
    (AkShare 版本) 查询中国的年度宏观经济指标（CPI 或 GDP），并返回最新的数据点及格式化后的文本。
    """
    logger.info(f"正在调用 AkShare 获取宏观数据: {indicator}")

    try:
        # 1. 数据获取和初始重命名 (保持不变)
        if indicator == 'GDP':
            df = ak.macro_china_gdp_yearly()# akshare获取数据
            df = df.rename(columns={'年度': '年份', '国内生产总值(亿元)': 'GDP(亿元)'})

        elif indicator == 'CPI':
            df = ak.macro_china_cpi_yearly()# akshare获取数据
            df = df.rename(columns={'年度': '年份', '居民消费价格指数(上年=100)': 'CPI_Index'})

        else:
            return json.dumps({"error": f"不支持的宏观经济指标: {indicator}。请查询 'CPI' 或 'GDP'。"})

        if df.empty:
            return json.dumps({"error": f"未找到 {indicator} 的数据 (AkShare)。"})

        # 取最新的5条数据进行格式化
        df_latest = df.head(5).reset_index(drop=True)

        # 2. **关键修改：生成格式化字符串**
        format_lines = []

        if indicator == 'GDP':
            for index, row in df_latest.iterrows():
                line = f"年份 {row['年份']}: 国内生产总值(GDP) {row['GDP(亿元)']} 亿元"
                format_lines.append(line)

        elif indicator == 'CPI':
            for index, row in df_latest.iterrows():
                # 假设 CPI_Index 的值是指数，LLM 需要知道它是基期=100的指数
                line = f"年份 {row['年份']}: 居民消费价格指数(上年=100) 为 {row['CPI_Index']}"
                format_lines.append(line)

        # 将所有行合并成一个字符串
        formatted_string = f"最近的 {indicator} 宏观数据显示：\n" + "\n".join(format_lines)

        # 转换为 JSON 数组格式 (保持不变)
        data_list = df_latest.to_dict(orient='records')

        # 更新 summary 结构
        summary = {
            "indicator": indicator,
            "latest_data_points": data_list,  # 保留原始 JSON 数据以备 Agent 复杂分析
            "formatted_output": formatted_string
        }

        logger.info(f"AkShare 宏观数据 {indicator} 获取成功。")
        return json.dumps(summary, ensure_ascii=False)

    except Exception as e:
        logger.error(f"AkShare 宏观数据调用失败: {e}")
        return json.dumps({"error": f"AkShare 宏观数据调用失败: {e}"})

AVAILABLE_TOOLS = {
    "get_stock_zh_a_spot_data": get_stock_zh_a_spot_data,
    "visualize_stock_data_trend": visualize_stock_data_trend,
    "query_macro_data":query_macro_data # 添加新工具
}