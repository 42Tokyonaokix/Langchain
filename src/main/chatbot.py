"""LangChain Agent Chatbot"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.prebuilt import create_react_agent

# 環境変数の読み込み
load_dotenv()

# ツールの定義
@tool
def calculator(expression: str) -> str:
    """数式を計算します。四則演算や累乗などの数学的な計算に使用してください。

    Args:
        expression: 計算する数式（例: "2 + 3 * 4", "10 ** 2"）
    """
    try:
        # 安全な計算のため、許可された文字のみを使用
        allowed_chars = set("0123456789+-*/.() ")
        if not all(c in allowed_chars for c in expression):
            return "エラー: 許可されていない文字が含まれています"
        result = eval(expression)
        return f"{expression} = {result}"
    except Exception as e:
        return f"計算エラー: {e}"


@tool
def get_current_time() -> str:
    """現在の日時を取得します。"""
    from datetime import datetime
    now = datetime.now()
    return now.strftime("%Y年%m月%d日 %H:%M:%S")


@tool
def search_yakkan(query: str) -> str:
    """電力関連の資料を検索します。約款、システムマニュアル、Q&A、契約情報などを検索できます。

    Args:
        query: 検索キーワード（例: "契約期間", "解約", "届出", "インバランス", "操作方法"）
    """
    from src.utils.rag import search_documents
    return search_documents(query, k=5)


def create_agent():
    """エージェントを作成"""
    model_name = os.getenv("OPENAI_MODEL", "gpt-4o")

    llm = ChatOpenAI(
        model=model_name,
        temperature=0.7,
    )

    tools = [calculator, get_current_time, search_yakkan]

    system_prompt = """あなたはデジタルグリッド株式会社の電力契約に関するカスタマーサポートAIです。

ユーザーからの質問に対しては、必ず search_yakkan ツールを使って関連情報を検索し、
その情報に基づいて回答してください。

検索対象には以下が含まれます：
- 電気需給約款（低圧・高圧）
- 送配電事業者の託送供給等約款
- システムマニュアル
- Q&Aリスト
- その他の契約関連資料

回答する際は：
- 検索結果の内容を正確に伝える
- わかりやすい言葉で説明する
- 不明な点は「資料に記載がないため、お客様センターにお問い合わせください」と案内する

日本語で応答してください。"""

    return create_react_agent(llm, tools, prompt=system_prompt)


def main():
    """メインループ"""
    print("=" * 50)
    print("LangChain Agent Chatbot")
    print("終了するには 'quit' または 'exit' と入力してください")
    print("=" * 50)

    agent = create_agent()
    messages = []

    while True:
        try:
            user_input = input("\nYou: ").strip()

            if not user_input:
                continue

            if user_input.lower() in ["quit", "exit", "q"]:
                print("さようなら！")
                break

            # メッセージを追加
            messages.append(HumanMessage(content=user_input))

            # エージェントを実行
            result = agent.invoke({"messages": messages})

            # 最後のAIメッセージを取得
            ai_message = result["messages"][-1]
            response = ai_message.content
            print(f"\nAssistant: {response}")

            # 会話履歴を更新
            messages.append(ai_message)

        except KeyboardInterrupt:
            print("\n\n中断されました。さようなら！")
            break
        except Exception as e:
            print(f"\nエラーが発生しました: {e}")


if __name__ == "__main__":
    main()
