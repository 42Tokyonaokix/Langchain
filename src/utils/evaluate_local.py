"""
ローカル評価スクリプト（LangSmith不要）

test_cases.csv を読み込んでチャットボットの精度を評価し、
結果をCSVファイルに出力します。
"""

import csv
import time
import os
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

load_dotenv()

# パス設定
PROJECT_ROOT = Path(__file__).parent.parent.parent
TEST_CASES_PATH = PROJECT_ROOT / "test_cases.csv"
RESULTS_DIR = PROJECT_ROOT / "evaluation_results"

# 評価用LLM（コスト削減のためgpt-4o-miniを使用）
EVAL_MODEL = "gpt-4o-mini"


def load_test_cases() -> list[dict]:
    """テストケースを読み込む"""
    cases = []
    with open(TEST_CASES_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            question = row.get("質問", row.get("question", "")).strip()
            answer = row.get("回答", row.get("answer", "")).strip()
            category = row.get("カテゴリ", row.get("category", "")).strip()

            if question and answer:
                cases.append({
                    "question": question,
                    "expected": answer,
                    "category": category,
                })
    return cases


def run_chatbot(question: str) -> dict:
    """チャットボットを実行して回答とソースを取得"""
    from src.main.chatbot import create_agent
    from langchain_core.messages import ToolMessage

    agent = create_agent()
    result = agent.invoke({"messages": [HumanMessage(content=question)]})

    # 最終回答
    answer = result["messages"][-1].content

    # ToolMessage（検索結果）を抽出
    sources = []
    for msg in result["messages"]:
        if isinstance(msg, ToolMessage):
            sources.append(msg.content)

    return {
        "answer": answer,
        "sources": "\n---\n".join(sources) if sources else ""
    }


def evaluate_answer(question: str, expected: str, actual: str, llm: ChatOpenAI) -> dict:
    """LLMを使って回答を評価"""
    eval_prompt = f"""以下の質問に対する「予測回答」が「期待回答」と意味的に一致しているか評価してください。

【質問】
{question}

【期待回答】
{expected}

【予測回答】
{actual}

【評価基準】
- 主要な情報（数値、用語、結論）が含まれているか
- 事実として正しいか
- 意味的に同等か

【出力形式】
以下の形式で出力してください：
スコア: 0.0〜1.0の数値
理由: 1行で簡潔に

例:
スコア: 0.8
理由: 主要な情報は含まれているが、一部詳細が不足"""

    result = llm.invoke(eval_prompt)
    response = result.content

    # スコアを抽出
    score = 0.0
    reason = ""
    for line in response.split("\n"):
        if line.startswith("スコア:") or line.startswith("スコア："):
            try:
                score_str = line.split(":", 1)[1].strip()
                score = float(score_str)
                score = max(0.0, min(1.0, score))
            except:
                pass
        elif line.startswith("理由:") or line.startswith("理由："):
            reason = line.split(":", 1)[1].strip()

    return {"score": score, "reason": reason}


def run_evaluation(max_cases: int = None, output_name: str = None):
    """評価を実行"""
    print("=" * 60)
    print("ローカル評価を開始します")
    print("=" * 60)

    # テストケース読み込み
    test_cases = load_test_cases()
    if max_cases:
        test_cases = test_cases[:max_cases]
    print(f"テストケース: {len(test_cases)}件")

    # 評価用LLM
    eval_llm = ChatOpenAI(model=EVAL_MODEL, temperature=0)

    # 結果格納
    results = []
    scores = []

    # 出力ディレクトリ作成
    RESULTS_DIR.mkdir(exist_ok=True)

    # 評価実行
    for i, case in enumerate(test_cases, 1):
        print(f"\n[{i}/{len(test_cases)}] {case['question'][:50]}...")

        start_time = time.time()

        try:
            # チャットボット実行
            chatbot_result = run_chatbot(case["question"])
            actual = chatbot_result["answer"]
            sources = chatbot_result["sources"]
            latency = time.time() - start_time

            # 評価
            eval_result = evaluate_answer(
                case["question"], case["expected"], actual, eval_llm
            )

            result = {
                "id": i,
                "category": case["category"],
                "question": case["question"],
                "expected": case["expected"],
                "actual": actual,
                "sources": sources,
                "score": eval_result["score"],
                "reason": eval_result["reason"],
                "latency": round(latency, 2),
                "status": "success",
            }

            scores.append(eval_result["score"])
            print(f"  スコア: {eval_result['score']:.1f} | {eval_result['reason'][:40]}")

        except Exception as e:
            result = {
                "id": i,
                "category": case["category"],
                "question": case["question"],
                "expected": case["expected"],
                "actual": "",
                "sources": "",
                "score": 0.0,
                "reason": f"エラー: {str(e)}",
                "latency": 0,
                "status": "error",
            }
            print(f"  エラー: {e}")

        results.append(result)

    # 結果サマリー
    print("\n" + "=" * 60)
    print("評価完了")
    print("=" * 60)

    if scores:
        avg_score = sum(scores) / len(scores)
        high = len([s for s in scores if s >= 0.9])
        mid = len([s for s in scores if 0.7 <= s < 0.9])
        low = len([s for s in scores if s < 0.7])

        print(f"平均スコア: {avg_score:.2%}")
        print(f"スコア分布:")
        print(f"  0.9以上: {high}件 ({high/len(scores):.1%})")
        print(f"  0.7-0.9: {mid}件 ({mid/len(scores):.1%})")
        print(f"  0.7未満: {low}件 ({low/len(scores):.1%})")
        print(f"正解率 (>=0.7): {high+mid}件 ({(high+mid)/len(scores):.1%})")

    # CSV出力
    if output_name:
        filename = f"{output_name}.csv"
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"eval_{timestamp}.csv"

    output_path = RESULTS_DIR / filename

    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "id", "category", "question", "expected", "actual", "sources",
            "score", "reason", "latency", "status"
        ])
        writer.writeheader()
        writer.writerows(results)

    print(f"\n結果を保存しました: {output_path}")

    return results


def main():
    """メイン関数"""
    import sys

    # コマンドライン引数
    max_cases = None
    output_name = None

    for arg in sys.argv[1:]:
        if arg.startswith("--max="):
            max_cases = int(arg.split("=")[1])
        elif arg.startswith("--output="):
            output_name = arg.split("=")[1]

    run_evaluation(max_cases=max_cases, output_name=output_name)


if __name__ == "__main__":
    main()
