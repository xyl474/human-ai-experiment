# human_ai_game_platform.py
import streamlit as st
import pandas as pd
import numpy as np
import json
import time
import requests
import random
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import openai
from openai import OpenAI
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64
import uuid

# ===================== 页面配置 =====================
st.set_page_config(
    page_title="人类-AI博弈实验平台",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===================== API配置 =====================
# 从环境变量或secrets获取API密钥
DEEPSEEK_API_KEY = st.secrets.get("DEEPSEEK_API_KEY", "")
DOUBAO_API_KEY = st.secrets.get("DOUBAO_API_KEY", "")

# ===================== 初始化客户端 =====================
deepseek_client = None
if DEEPSEEK_API_KEY:
    deepseek_client = OpenAI(
        api_key=DEEPSEEK_API_KEY,
        base_url="https://api.deepseek.com"
    )

doubao_client = None
if DOUBAO_API_KEY:
    doubao_client = OpenAI(
        api_key=DOUBAO_API_KEY,
        base_url="https://ark.cn-beijing.volces.com/api/v3"
    )


# ===================== 实验参数 =====================
class ExperimentConfig:
    # 实验设计
    NUM_ROUNDS = 10  # 每个实验10轮
    REWARD_MATRIX_PD = {  # 囚徒困境收益矩阵
        ("J", "J"): (8, 8),  # 双方合作
        ("J", "F"): (0, 10),  # 我被背叛
        ("F", "J"): (10, 0),  # 我背叛对方
        ("F", "F"): (5, 5)  # 双方背叛
    }

    REWARD_MATRIX_BOS = {  # 性别之战收益矩阵
        ("J", "J"): (10, 7),  # 协调在J，玩家1得高分
        ("F", "F"): (7, 10),  # 协调在F，玩家2得高分
        ("J", "F"): (0, 0),  # 不匹配
        ("F", "J"): (0, 0)  # 不匹配
    }

    # AI模型配置
    AI_MODELS = ["DeepSeek", "Doubao"]
    MODEL_VERSIONS = ["Standard", "SCoT"]

    # 实验条件
    EXPERIMENT_CONDITIONS = [
        {"model": "DeepSeek", "version": "Standard", "game": "PrisonerDilemma"},
        {"model": "DeepSeek", "version": "SCoT", "game": "PrisonerDilemma"},
        {"model": "DeepSeek", "version": "Standard", "game": "BattleOfSexes"},
        {"model": "DeepSeek", "version": "SCoT", "game": "BattleOfSexes"},
        {"model": "Doubao", "version": "Standard", "game": "PrisonerDilemma"},
        {"model": "Doubao", "version": "SCoT", "game": "PrisonerDilemma"},
        {"model": "Doubao", "version": "Standard", "game": "BattleOfSexes"},
        {"model": "Doubao", "version": "SCoT", "game": "BattleOfSexes"},
    ]


# ===================== 工具函数 =====================
def calculate_points(game_type: str, action1: str, action2: str) -> Tuple[int, int]:
    """根据游戏类型计算收益"""
    if game_type == "PrisonerDilemma":
        return ExperimentConfig.REWARD_MATRIX_PD.get((action1, action2), (0, 0))
    else:  # BattleOfSexes
        return ExperimentConfig.REWARD_MATRIX_BOS.get((action1, action2), (0, 0))


def get_game_description(game_type: str) -> str:
    """获取游戏描述"""
    if game_type == "PrisonerDilemma":
        return """**囚徒困境规则**：
- 如果你和对方都选择 J（合作）：各得8分
- 如果你选J（合作），对方选F（背叛）：你得0分，对方得10分
- 如果你选F（背叛），对方选J（合作）：你得10分，对方得0分
- 如果双方都选F（背叛）：各得5分

**你的目标**：在10轮游戏中最大化自己的总得分"""
    else:  # BattleOfSexes
        return """**性别之战规则**：
- 如果双方都选择 J：你得10分，对方得7分
- 如果双方都选择 F：你得7分，对方得10分
- 如果选择不同：双方都得0分

**你的目标**：在10轮游戏中最大化自己的总得分"""


# ===================== AI调用函数 =====================
def get_scot_prompt(game_type: str, history: List[Dict], current_round: int,
                    model_name: str, version: str) -> str:
    """生成社会思维链提示词"""
    if version == "Standard":
        # 标准提示词
        if game_type == "PrisonerDilemma":
            return f"""你正在与一个人类玩家进行重复囚徒困境游戏，共10轮。

游戏规则：
- 双方都选J：各得8分
- 你选J，对方选F：你得0分，对方得10分
- 你选F，对方选J：你得10分，对方得0分
- 双方都选F：各得5分

历史记录：
{format_history(history)}

这是第{current_round}轮。请从J和F中选择一个，回复单个字母：J 或 F"""
        else:  # BattleOfSexes
            return f"""你正在与一个人类玩家进行性别之战游戏，共10轮。

游戏规则：
- 双方都选J：你得10分，对方得7分
- 双方都选F：你得7分，对方得10分
- 选择不同：都得0分

历史记录：
{format_history(history)}

这是第{current_round}轮。请从J和F中选择一个，回复单个字母：J 或 F"""
    else:  # SCoT版本
        # 社会思维链提示词
        if game_type == "PrisonerDilemma":
            return f"""你正在与一个人类玩家进行重复囚徒困境游戏，共10轮。

请按以下步骤进行社会思维链推理：

1. **社会推理**：考虑对方的感受和意图。从历史记录看，对方是怎样的玩家？

2. **互惠分析**：对方之前如何对待你？你是否应该以牙还牙，还是表现出宽容？

3. **关系建设**：你想与对方建立什么样的关系？竞争还是合作？

4. **长期视角**：这是第{current_round}/10轮。考虑剩余轮次，你的策略应该如何调整？

5. **情感预测**：你的选择会让对方产生什么感受？这会影响后续互动吗？

6. **规范思考**：在这种情境下，大多数人会如何选择？社会规范是什么？

历史记录：
{format_history(history)}

基于以上思考，第{current_round}轮你选择：J 或 F

请只回复一个字母：J 或 F"""
        else:  # BattleOfSexes
            return f"""你正在与一个人类玩家进行性别之战游戏，共10轮。

请按以下步骤进行社会思维链推理：

1. **协调分析**：我们需要协调选择才能得分。对方倾向于选择J还是F？

2. **公平考虑**：协调在J时你得10分对方得7分，协调在F时你得7分对方得10分。如何平衡？

3. **轮流策略**：是否应该轮流让对方得高分，以建立互惠关系？

4. **信号解读**：对方的历史选择传递了什么信号？你想传递什么信号？

5. **长期关系**：你想建立什么样的协调模式？固定的还是轮流的？

历史记录：
{format_history(history)}

基于以上思考，第{current_round}轮你选择：J 或 F

请只回复一个字母：J 或 F"""


def format_history(history: List[Dict]) -> str:
    """格式化历史记录"""
    if not history:
        return "这是第一轮，暂无历史记录。"

    text = ""
    for i, record in enumerate(history, 1):
        text += f"第{i}轮: 你选择{record['ai_action']}, 对方选择{record['human_action']}. "
        text += f"你获得{record['ai_points']}分, 对方获得{record['human_points']}分。\n"
    return text


def call_ai_model(model_name: str, version: str, prompt: str) -> str:
    """调用AI模型"""
    try:
        if model_name == "DeepSeek" and deepseek_client:
            response = deepseek_client.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=1
            )
            return response.choices[0].message.content.strip().upper()

        elif model_name == "Doubao" and doubao_client:
            response = doubao_client.chat.completions.create(
                model="doubao-1.5-32k",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=1
            )
            return response.choices[0].message.content.strip().upper()

        return random.choice(['J', 'F'])  # 失败时随机选择

    except Exception as e:
        st.error(f"调用AI模型失败: {str(e)[:100]}")
        return random.choice(['J', 'F'])


# ===================== 数据管理 =====================
class DataManager:
    def __init__(self):
        if 'participant_id' not in st.session_state:
            st.session_state.participant_id = str(uuid.uuid4())[:8]

        if 'experiment_data' not in st.session_state:
            st.session_state.experiment_data = []

        if 'current_condition' not in st.session_state:
            st.session_state.current_condition = 0

        if 'current_round' not in st.session_state:
            st.session_state.current_round = 1

        if 'condition_data' not in st.session_state:
            st.session_state.condition_data = {}

    def save_round_data(self, condition_idx: int, round_num: int,
                        human_action: str, ai_action: str,
                        human_points: int, ai_points: int,
                        decision_time: float):
        """保存一轮数据"""
        condition = ExperimentConfig.EXPERIMENT_CONDITIONS[condition_idx]

        round_data = {
            'participant_id': st.session_state.participant_id,
            'condition_index': condition_idx,
            'model': condition['model'],
            'version': condition['version'],
            'game': condition['game'],
            'round': round_num,
            'human_action': human_action,
            'ai_action': ai_action,
            'human_points': human_points,
            'ai_points': ai_points,
            'total_human_points': self._calculate_total(condition_idx, 'human'),
            'total_ai_points': self._calculate_total(condition_idx, 'ai'),
            'decision_time': decision_time,
            'timestamp': datetime.now().isoformat()
        }

        st.session_state.experiment_data.append(round_data)

        # 保存到条件数据
        key = f"{condition['model']}_{condition['version']}_{condition['game']}"
        if key not in st.session_state.condition_data:
            st.session_state.condition_data[key] = []

        st.session_state.condition_data[key].append(round_data)

    def _calculate_total(self, condition_idx: int, player: str) -> int:
        """计算当前条件下的总得分"""
        total = 0
        for data in st.session_state.experiment_data:
            if data['condition_index'] == condition_idx:
                total += data[f'{player}_points']
        return total

    def get_condition_history(self, condition_idx: int) -> List[Dict]:
        """获取指定条件的历史记录"""
        history = []
        for data in st.session_state.experiment_data:
            if data['condition_index'] == condition_idx:
                history.append({
                    'round': data['round'],
                    'human_action': data['human_action'],
                    'ai_action': data['ai_action'],
                    'human_points': data['human_points'],
                    'ai_points': data['ai_points']
                })
        return sorted(history, key=lambda x: x['round'])

    def get_summary_data(self) -> pd.DataFrame:
        """获取汇总数据"""
        if not st.session_state.experiment_data:
            return pd.DataFrame()

        df = pd.DataFrame(st.session_state.experiment_data)
        return df

    def export_data(self) -> str:
        """导出数据为JSON"""
        return json.dumps(st.session_state.experiment_data, ensure_ascii=False, indent=2)


# ===================== 实验界面 =====================
def show_welcome():
    """显示欢迎页面"""
    st.title("🧠 人类-AI博弈实验")

    st.markdown("""
    ## 欢迎参加人类与AI的博弈实验！

    ### 实验目的
    本研究旨在探索人类与人工智能在策略性互动中的行为模式，并测试“社会思维链”干预能否提升AI的博弈表现。

    ### 实验流程
    1. **基本信息登记**：填写一些基本信息
    2. **实验说明**：了解两种博弈游戏的规则
    3. **练习阶段**：与AI进行简短练习
    4. **正式实验**：与不同AI进行囚徒困境和性别之战
    5. **问卷调查**：分享你的体验和策略
    6. **结果展示**：查看你的表现和分析报告

    ### 实验时长
    - 预计耗时：20-30分钟
    - 实验轮次：8个条件 × 10轮 = 80轮决策

    ### 您的贡献
    - 您的参与将为人类与AI的互动研究提供宝贵数据。
    - 实验结束后，您将看到自己完整的行为表现分析报告。

    ### 隐私保护
    - 所有数据将匿名处理
    - 仅用于学术研究
    - 可随时退出实验

    ---
    """)

    if st.button("开始实验", type="primary", use_container_width=True):
        st.session_state.current_page = "demographics"
        st.rerun()




def show_demographics():
    """显示人口学信息页面"""
    st.title("📋 基本信息登记")

    with st.form("demographics_form"):
        col1, col2 = st.columns(2)

        with col1:
            age = st.number_input("年龄", min_value=18, max_value=80, value=25)
            gender = st.selectbox("性别", ["男", "女", "其他", "不愿透露"])

        with col2:
            education = st.selectbox("最高学历",
                                     ["高中及以下", "大专", "本科", "硕士", "博士"])
            major = st.text_input("专业/领域")

        game_experience = st.slider("博弈游戏经验（1-完全没有经验，7-经验丰富）", 1, 7, 4)
        ai_familiarity = st.slider("对AI的了解程度（1-完全不了解，7-非常了解）", 1, 7, 4)

        submitted = st.form_submit_button("下一步", use_container_width=True)

        if submitted:
            st.session_state.demographics = {
                'age': age,
                'gender': gender,
                'education': education,
                'major': major,
                'game_experience': game_experience,
                'ai_familiarity': ai_familiarity
            }
            st.session_state.current_page = "instructions"
            st.rerun()


def show_instructions():
    """显示实验说明"""
    st.title("📖 实验说明")

    tab1, tab2 = st.tabs(["囚徒困境", "性别之战"])

    with tab1:
        st.markdown(get_game_description("PrisonerDilemma"))

        st.markdown("""
        ### 收益矩阵
        """)

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("双方合作", "各得8分", delta=None)
        with col2:
            st.metric("你合作，AI背叛", "你得0分，AI得10分", delta="-8", delta_color="inverse")
        with col3:
            st.metric("你背叛，AI合作", "你得10分，AI得0分", delta="+10")
        with col4:
            st.metric("双方背叛", "各得5分", delta=None)

    with tab2:
        st.markdown(get_game_description("BattleOfSexes"))

        st.markdown("""
        ### 收益矩阵
        """)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("都选J", "你得10分，AI得7分", delta="+3", delta_color="off")
        with col2:
            st.metric("都选F", "你得7分，AI得10分", delta="-3", delta_color="inverse")
        with col3:
            st.metric("选择不同", "都得0分", delta="-10", delta_color="inverse")

    st.markdown("---")

    st.markdown("""
    ### AI对手说明

    你将与4种不同的AI对手进行游戏：

    1. **DeepSeek标准版**：使用标准提示词的AI
    2. **DeepSeek社会思维链版**：使用社会推理提示词的AI
    3. **豆包标准版**：使用标准提示词的AI
    4. **豆包社会思维链版**：使用社会推理提示词的AI

    **社会思维链**：AI会进行多步社会推理，考虑对方感受、长期关系、社会规范等因素。
    """)

    if st.button("开始练习", type="primary", use_container_width=True):
        st.session_state.current_page = "practice"
        st.rerun()


def show_practice():
    """显示练习阶段"""
    st.title("🎮 练习阶段")

    st.info("""
    **练习说明**：
    - 你将与一个简单的AI进行5轮囚徒困境练习
    - 目的是熟悉界面和操作
    - 练习结果不计入最终得分
    """)

    if 'practice_data' not in st.session_state:
        st.session_state.practice_data = []
        st.session_state.practice_round = 1
        st.session_state.practice_history = []
        st.session_state.practice_human_total = 0
        st.session_state.practice_ai_total = 0

    if st.session_state.practice_round <= 5:
        st.subheader(f"练习第 {st.session_state.practice_round}/5 轮")

        col1, col2 = st.columns(2)

        with col1:
            if st.button("J（合作）", use_container_width=True, type="primary"):
                human_action = "J"
                process_practice_round(human_action)

        with col2:
            if st.button("F（背叛）", use_container_width=True, type="secondary"):
                human_action = "F"
                process_practice_round(human_action)

             # 显示历史记录
                if st.session_state.practice_history:
                    st.subheader("历史记录")
                    # 确保使用正确的列名
                    history_df = pd.DataFrame(st.session_state.practice_history)
                    # 重命名列以便更友好地显示
                    display_df = history_df.copy()
                    display_df.columns = ["轮次", "你的选择", "AI选择", "你的得分", "AI得分"]
                    st.dataframe(display_df, use_container_width=True)

            # 显示当前得分
            col1, col2 = st.columns(2)
            with col1:
                st.metric("你的得分", st.session_state.practice_human_total)
            with col2:
                st.metric("AI得分", st.session_state.practice_ai_total)
    else:
        st.success("练习完成！")

        # 显示练习总结
        col1, col2 = st.columns(2)
        with col1:
            st.metric("你的总得分", st.session_state.practice_human_total)
        with col2:
            st.metric("AI总得分", st.session_state.practice_ai_total)

        if st.button("开始正式实验", type="primary", use_container_width=True):
            st.session_state.current_page = "experiment"
            st.rerun()


def process_practice_round(human_action: str):
    """处理练习轮次"""
    # 简单的AI策略：以牙还牙
    if not st.session_state.practice_history:
        ai_action = "J"  # 第一轮合作
    else:
        # 从历史记录中获取上次的人类选择
        last_record = st.session_state.practice_history[-1]
        # 检查使用的是中文键名还是英文键名
        if '你的选择' in last_record:
            last_human_action = last_record['你的选择']
        elif 'human_action' in last_record:
            last_human_action = last_record['human_action']
        else:
            # 如果没有找到，默认合作
            last_human_action = "J"
        ai_action = last_human_action  # 以牙还牙

    # 计算得分
    human_points, ai_points = calculate_points("PrisonerDilemma", human_action, ai_action)

    # 更新总分
    st.session_state.practice_human_total += human_points
    st.session_state.practice_ai_total += ai_points

    # 保存记录 - 使用统一的数据结构
    st.session_state.practice_history.append({
        "轮次": st.session_state.practice_round,
        "你的选择": human_action,
        "AI选择": ai_action,
        "你的得分": human_points,
        "AI得分": ai_points
    })

    st.session_state.practice_round += 1
    st.rerun()

def show_experiment():
    """显示实验主界面"""
    data_manager = DataManager()
    conditions = ExperimentConfig.EXPERIMENT_CONDITIONS

    if st.session_state.current_condition < len(conditions):
        current_condition = conditions[st.session_state.current_condition]

        # 显示当前条件信息
        st.title(f"🎮 实验阶段 {st.session_state.current_condition + 1}/8")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("AI模型", current_condition['model'])
        with col2:
            st.metric("AI版本", current_condition['version'])
        with col3:
            game_name = "囚徒困境" if current_condition['game'] == "PrisonerDilemma" else "性别之战"
            st.metric("游戏类型", game_name)

        st.markdown("---")

        # 显示游戏说明
        st.markdown(get_game_description(current_condition['game']))

        st.markdown("---")

        # 显示当前轮次
        st.subheader(f"第 {st.session_state.current_round}/{ExperimentConfig.NUM_ROUNDS} 轮")

        # 决策按钮
        col1, col2 = st.columns(2)

        start_time = time.time()

        with col1:
            if st.button("J", use_container_width=True, type="primary",
                         help="选择J（合作/足球）"):
                process_experiment_round("J", current_condition, start_time, data_manager)

        with col2:
            if st.button("F", use_container_width=True, type="secondary",
                         help="选择F（背叛/芭蕾）"):
                process_experiment_round("F", current_condition, start_time, data_manager)

        # 显示当前条件的历史记录
        history = data_manager.get_condition_history(st.session_state.current_condition)
        if history:
            st.subheader("当前游戏历史")
            history_df = pd.DataFrame(history)
            st.dataframe(history_df, use_container_width=True)

            # 显示当前得分
            human_total = sum([h['human_points'] for h in history])
            ai_total = sum([h['ai_points'] for h in history])

            col1, col2 = st.columns(2)
            with col1:
                st.metric("你的当前得分", human_total)
            with col2:
                st.metric("AI当前得分", ai_total)

    else:
        st.success("🎉 实验完成！")
        st.session_state.current_page = "questionnaire"
        st.rerun()


def process_experiment_round(human_action: str, current_condition: Dict,
                             start_time: float, data_manager: DataManager):
    """处理实验轮次"""
    # 获取历史记录
    history = data_manager.get_condition_history(st.session_state.current_condition)

    # 调用AI
    prompt = get_scot_prompt(
        current_condition['game'],
        history,
        st.session_state.current_round,
        current_condition['model'],
        current_condition['version']
    )

    with st.spinner(f"等待{current_condition['model']} {current_condition['version']}做出决策..."):
        ai_action = call_ai_model(
            current_condition['model'],
            current_condition['version'],
            prompt
        )

    # 计算决策时间
    decision_time = time.time() - start_time

    # 计算得分
    human_points, ai_points = calculate_points(
        current_condition['game'],
        human_action,
        ai_action
    )

    # 保存数据
    data_manager.save_round_data(
        st.session_state.current_condition,
        st.session_state.current_round,
        human_action,
        ai_action,
        human_points,
        ai_points,
        decision_time
    )

    # 显示结果
    st.success(f"**结果**：你选择{human_action}，AI选择{ai_action}")
    st.info(f"**得分**：你获得{human_points}分，AI获得{ai_points}分")

    # 更新轮次
    st.session_state.current_round += 1

    # 检查是否完成当前条件
    if st.session_state.current_round > ExperimentConfig.NUM_ROUNDS:
        st.session_state.current_round = 1
        st.session_state.current_condition += 1

        if st.session_state.current_condition < len(ExperimentConfig.EXPERIMENT_CONDITIONS):
            st.info("完成当前条件，准备下一个条件...")
        else:
            st.success("所有实验条件完成！")

    st.rerun()


def show_questionnaire():
    """显示问卷调查"""
    st.title("📊 问卷调查")

    st.markdown("请回答以下问题，帮助我们更好地理解你的决策策略。")

    with st.form("questionnaire_form"):
        # 策略相关问题
        st.subheader("策略评估")

        q1 = st.slider("1. 在囚徒困境中，你倾向于采取什么策略？\n(1=总是合作, 7=总是背叛)", 1, 7, 4)

        q2 = st.slider("2. 在性别之战中，你倾向于采取什么策略？\n(1=总是选J, 7=总是选F)", 1, 7, 4)

        q3 = st.slider("3. 你是否根据AI的行为调整自己的策略？\n(1=从不调整, 7=经常调整)", 1, 7, 4)

        # 对AI的感知
        st.subheader("对AI的感知")

        q4 = st.slider("4. 你觉得哪个AI版本更合作？\n(1=标准版更合作, 7=社会思维链版更合作)", 1, 7, 4)

        q5 = st.slider("5. 你觉得哪个AI模型更聪明？\n(1=DeepSeek更聪明, 7=豆包更聪明)", 1, 7, 4)

        q6 = st.text_area("6. 请描述你观察到AI的行为模式：")

        # 主观体验
        st.subheader("主观体验")

        q7 = st.slider("7. 实验过程的趣味性如何？\n(1=非常无聊, 7=非常有趣)", 1, 7, 4)

        q8 = st.slider("8. 实验界面的易用性如何？\n(1=非常难用, 7=非常易用)", 1, 7, 4)

        q9 = st.text_area("9. 其他意见或建议：")

        submitted = st.form_submit_button("提交问卷", use_container_width=True)

        if submitted:
            st.session_state.questionnaire = {
                'q1': q1, 'q2': q2, 'q3': q3,
                'q4': q4, 'q5': q5, 'q6': q6,
                'q7': q7, 'q8': q8, 'q9': q9
            }
            st.session_state.current_page = "results"
            st.rerun()


def show_results():
    """显示实验结果"""
    st.title("📈 实验结果")

    data_manager = DataManager()
    df = data_manager.get_summary_data()

    if df.empty:
        st.warning("暂无实验数据")
        return

    # 总体统计
    st.subheader("总体表现")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        total_human_score = df['human_points'].sum()
        st.metric("你的总得分", total_human_score)

    with col2:
        total_ai_score = df['ai_points'].sum()
        st.metric("AI总得分", total_ai_score)

    with col3:
        avg_decision_time = df['decision_time'].mean()
        st.metric("平均决策时间", f"{avg_decision_time:.2f}秒")

    with col4:
        total_cooperation = len(df[df['human_action'] == 'J'])
        cooperation_rate = total_cooperation / len(df) * 100
        st.metric("你的合作率", f"{cooperation_rate:.1f}%")

    # 详细分析
    st.markdown("---")
    st.subheader("详细分析")

    # 按条件分析
    conditions = ExperimentConfig.EXPERIMENT_CONDITIONS

    for i, condition in enumerate(conditions):
        condition_df = df[df['condition_index'] == i]

        if not condition_df.empty:
            st.markdown(f"**{condition['model']} {condition['version']} - {condition['game']}**")

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                human_score = condition_df['human_points'].sum()
                st.metric("你的得分", human_score)

            with col2:
                ai_score = condition_df['ai_points'].sum()
                st.metric("AI得分", ai_score)

            with col3:
                if condition['game'] == "PrisonerDilemma":
                    human_coop = len(condition_df[condition_df['human_action'] == 'J'])
                    coop_rate = human_coop / len(condition_df) * 100
                    st.metric("你的合作率", f"{coop_rate:.1f}%")
                else:
                    coordination = len(condition_df[
                                           condition_df['human_action'] == condition_df['ai_action']
                                           ])
                    coord_rate = coordination / len(condition_df) * 100
                    st.metric("协调成功率", f"{coord_rate:.1f}%")

            with col4:
                mutual_coop = len(condition_df[
                                      (condition_df['human_action'] == 'J') &
                                      (condition_df['ai_action'] == 'J')
                                      ])
                if condition['game'] == "PrisonerDilemma":
                    st.metric("互惠合作轮次", mutual_coop)
                else:
                    st.metric("协调在J轮次", mutual_coop)

    # 可视化
    st.markdown("---")
    st.subheader("数据可视化")

    tab1, tab2, tab3 = st.tabs(["得分对比", "行为模式", "决策时间"])

    with tab1:
        # 按模型和版本分组
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # 囚徒困境
        pd_df = df[df['game'] == 'PrisonerDilemma']
        if not pd_df.empty:
            pd_grouped = pd_df.groupby(['model', 'version']).agg({
                'human_points': 'sum',
                'ai_points': 'sum'
            }).reset_index()

            x = np.arange(len(pd_grouped))
            width = 0.35

            axes[0].bar(x - width / 2, pd_grouped['human_points'], width, label='人类', alpha=0.8)
            axes[0].bar(x + width / 2, pd_grouped['ai_points'], width, label='AI', alpha=0.8)
            axes[0].set_xlabel('模型-版本')
            axes[0].set_ylabel('总得分')
            axes[0].set_title('囚徒困境得分对比')
            axes[0].set_xticks(x)
            axes[0].set_xticklabels([f"{row['model']}\n{row['version']}" for _, row in pd_grouped.iterrows()])
            axes[0].legend()

        # 性别之战
        bos_df = df[df['game'] == 'BattleOfSexes']
        if not bos_df.empty:
            bos_grouped = bos_df.groupby(['model', 'version']).agg({
                'human_points': 'sum',
                'ai_points': 'sum'
            }).reset_index()

            x = np.arange(len(bos_grouped))
            width = 0.35

            axes[1].bar(x - width / 2, bos_grouped['human_points'], width, label='人类', alpha=0.8)
            axes[1].bar(x + width / 2, bos_grouped['ai_points'], width, label='AI', alpha=0.8)
            axes[1].set_xlabel('模型-版本')
            axes[1].set_ylabel('总得分')
            axes[1].set_title('性别之战得分对比')
            axes[1].set_xticks(x)
            axes[1].set_xticklabels([f"{row['model']}\n{row['version']}" for _, row in bos_grouped.iterrows()])
            axes[1].legend()

        plt.tight_layout()
        st.pyplot(fig)

    with tab2:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # 囚徒困境行为序列
        for i, condition in enumerate(conditions):
            if condition['game'] != 'PrisonerDilemma':
                continue

            condition_df = df[df['condition_index'] == i]
            if len(condition_df) >= 5:  # 至少5轮数据
                human_actions = condition_df['human_action'].values
                ai_actions = condition_df['ai_action'].values

                # 转换为数值
                human_numeric = [1 if a == 'J' else 0 for a in human_actions[:5]]
                ai_numeric = [1 if a == 'J' else 0 for a in ai_actions[:5]]

                axes[0].plot(range(1, 6), human_numeric, 'o-',
                             label=f"{condition['model']} {condition['version']} (人类)")
                axes[0].plot(range(1, 6), ai_numeric, 's--',
                             label=f"{condition['model']} {condition['version']} (AI)")

        axes[0].set_xlabel('轮次')
        axes[0].set_ylabel('行动 (1=J, 0=F)')
        axes[0].set_title('囚徒困境行为序列')
        axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[0].set_ylim(-0.1, 1.1)

        # 性别之战协调情况
        coordination_rates = []
        labels = []

        for i, condition in enumerate(conditions):
            if condition['game'] != 'BattleOfSexes':
                continue

            condition_df = df[df['condition_index'] == i]
            if not condition_df.empty:
                coordination = len(condition_df[
                                       condition_df['human_action'] == condition_df['ai_action']
                                       ])
                coord_rate = coordination / len(condition_df) * 100
                coordination_rates.append(coord_rate)
                labels.append(f"{condition['model']}\n{condition['version']}")

        if coordination_rates:
            axes[1].bar(range(len(coordination_rates)), coordination_rates)
            axes[1].set_xlabel('模型-版本')
            axes[1].set_ylabel('协调成功率 (%)')
            axes[1].set_title('性别之战协调成功率')
            axes[1].set_xticks(range(len(coordination_rates)))
            axes[1].set_xticklabels(labels, rotation=45, ha='right')

        plt.tight_layout()
        st.pyplot(fig)

    with tab3:
        fig, ax = plt.subplots(figsize=(10, 6))

        decision_times = []
        labels = []

        for i, condition in enumerate(conditions):
            condition_df = df[df['condition_index'] == i]
            if not condition_df.empty:
                avg_time = condition_df['decision_time'].mean()
                decision_times.append(avg_time)
                labels.append(f"{condition['model']}\n{condition['version']}\n{condition['game']}")

        if decision_times:
            bars = ax.bar(range(len(decision_times)), decision_times)
            ax.set_xlabel('实验条件')
            ax.set_ylabel('平均决策时间 (秒)')
            ax.set_title('各条件下的决策时间')
            ax.set_xticks(range(len(decision_times)))
            ax.set_xticklabels(labels, rotation=45, ha='right')

            # 添加数值标签
            for bar, time_val in zip(bars, decision_times):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2., height + 0.05,
                        f'{time_val:.2f}s', ha='center', va='bottom')

        plt.tight_layout()
        st.pyplot(fig)

    # 数据导出
    st.markdown("---")
    st.subheader("数据导出")

    col1, col2 = st.columns(2)

    with col1:
        # 导出CSV
        csv = df.to_csv(index=False).encode('utf-8-sig')
        st.download_button(
            label="下载CSV数据",
            data=csv,
            file_name=f"experiment_data_{st.session_state.participant_id}.csv",
            mime="text/csv",
            use_container_width=True
        )

    with col2:
        # 导出JSON
        json_data = data_manager.export_data()
        st.download_button(
            label="下载JSON数据",
            data=json_data,
            file_name=f"experiment_data_{st.session_state.participant_id}.json",
            mime="application/json",
            use_container_width=True
        )

    st.markdown("---")
    st.success("🎉 实验完成！感谢您的参与！")


# ===================== 主程序 =====================
def main():
    # 初始化session state
    if 'current_page' not in st.session_state:
        st.session_state.current_page = "welcome"

    # 页面路由
    pages = {
        "welcome": show_welcome,
        "demographics": show_demographics,
        "instructions": show_instructions,
        "practice": show_practice,
        "experiment": show_experiment,
        "questionnaire": show_questionnaire,
        "results": show_results
    }

    # 显示当前页面
    if st.session_state.current_page in pages:
        pages[st.session_state.current_page]()
    else:
        st.session_state.current_page = "welcome"
        st.rerun()


if __name__ == "__main__":
    main()