"""
LangSmith 評価スクリプト
データセットを使ってRAG/LLMの回答精度を評価します
"""
import os
from dotenv import load_dotenv
from langsmith import Client
from langsmith.evaluation import evaluate
from langchain_openai import ChatOpenAI
from src.utils.rag import search_documents

load_dotenv()

client = Client()

# 評価対象のLLM
llm = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o"), temperature=0)


def target(inputs: dict) -> dict:
    """評価対象の関数。質問を受け取り、RAGで検索して回答を返す"""
    question = inputs["question"]

    # RAGで約款を検索
    context = search_documents(question, k=3)

    # 検索結果を元にLLMが回答
    response = llm.invoke(f"""あなたは電力約款に関するカスタマーサポートAIです。
以下の参考情報を元に、質問に簡潔に回答してください。
参考情報にない内容は「約款に記載がないため、お客様センターにお問い合わせください」と回答してください。

【参考情報】
{context}

【質問】
{question}""")

    return {"answer": response.content}


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

    result = llm.invoke(eval_prompt)

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
    print("=" * 50)

    # 評価実行（並列実行するとChromaDBが競合するのでmax_concurrency=1）
    experiment_results = client.evaluate(
        target,
        data="electricity-yakkan-qa",
        evaluators=[correctness_evaluator],
        experiment_prefix="yakkan-rag-eval",
        max_concurrency=1,
    )

    print("\n評価完了！")
    print("LangSmithダッシュボードで詳細を確認してください")


if __name__ == "__main__":
    main()
