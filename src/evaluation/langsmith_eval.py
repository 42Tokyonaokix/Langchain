"""
LangSmith 評価スクリプト
データセットを使ってエージェント（Agentic Search + Web検索）の回答精度を評価します
"""
from langsmith import Client
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

from src.config import settings
from src.chatbot.agent import create_agent

client = Client()

# 評価対象のLLM（evaluator用）- 遅延初期化
_llm = None

def get_evaluator_llm():
    """評価用LLMを遅延初期化"""
    global _llm
    if _llm is None:
        _llm = ChatOpenAI(model=settings.OPENAI_MODEL, temperature=0)
    return _llm


def target(inputs: dict) -> dict:
    """評価対象の関数。エージェント（agentic_search + web_search）で回答"""
    question = inputs["question"]

    # エージェントを作成
    agent = create_agent()

    # エージェントを実行
    result = agent.invoke({"messages": [HumanMessage(content=question)]})

    # 最後のAIメッセージを取得
    ai_message = result["messages"][-1]

    return {"answer": ai_message.content}


def correctness_evaluator(outputs: dict, reference_outputs: dict) -> dict:
    """正確性を評価するevaluator"""
    predicted = outputs.get("answer", "")
    expected = reference_outputs.get("answer", "")

    # LLMを使って回答の正確性を評価
    eval_prompt = f"""以下の2つの回答を比較し、予測回答が期待回答と意味的に一致しているか評価してください。

期待回答: {expected}

予測回答: {predicted}

評価基準:
- 主要な情報が含まれているか
- 事実として正しいか
- 意味的に同等か

スコア (0.0-1.0) のみを数値で返してください。"""

    result = get_evaluator_llm().invoke(eval_prompt)

    try:
        score = float(result.content.strip())
        score = max(0.0, min(1.0, score))  # 0-1にクランプ
    except ValueError:
        score = 0.0

    return {
        "key": "correctness",
        "score": score,
    }


def main():
    print("=" * 50)
    print("LangSmith 評価を開始します")
    print("評価モード: Agentic Search（web_search + agentic_search）")
    print("=" * 50)

    prefix = "yakkan-agentic-eval"

    # 評価実行（並列実行するとChromaDBが競合するのでmax_concurrency=1）
    experiment_results = client.evaluate(
        target,
        data="electricity-yakkan-qa",
        evaluators=[correctness_evaluator],
        experiment_prefix=prefix,
        max_concurrency=1,
    )

    print("\n評価完了！")
    print("LangSmithダッシュボードで詳細を確認してください")


if __name__ == "__main__":
    main()
