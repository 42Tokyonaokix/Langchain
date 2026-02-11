"""
高速チャットボット - 5秒以内の応答を目標
"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent

load_dotenv()


@tool
def search_manual(query: str) -> str:
    """顧客ページのシステムマニュアル・約款を検索します。

    検索対象:
    - 電気需給約款（契約条件、料金体系）
    - 託送供給等約款
    - DGPサービス利用規約
    - システムマニュアル
    - 重要事項説明書

    Args:
        query: 検索キーワード
    """
    from src.utils.hybrid_rag import search_with_context
    return search_with_context(query, k=5)


def create_agent():
    """エージェントを作成"""
    llm = ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-4o"),
        temperature=0.3,
    )

    tools = [search_manual]

    system_prompt = """あなたはデジタルグリッド株式会社の顧客ポータル「DGM（Digital Grid Manager）」内に設置されたカスタマーサポートAIです。

【重要な前提】
- ユーザーは今まさにDGMの画面を操作しています
- システムマニュアルは、このDGMの操作説明書です
- 画面操作やボタンに関する質問は、システムマニュアルの情報を優先してください

【検索対象ドキュメント】
- システムマニュアル: DGMの画面操作、メニュー、ボタンの説明
- 電気需給約款: 契約条件、料金計算、支払い
- 託送供給等約款: 各電力会社エリアの技術基準・料金
- DGP利用規約: プラットフォーム利用条件、データ取扱い
- Q&Aリスト: よくある質問と回答

【回答のルール】
- 必ずsearch_manualで検索してから回答する
- ドキュメントにない情報は推測せず「カスタマーサポートにお問い合わせください」と案内
- 検索結果に「旧」「改正前」などの記載がある場合は、最新の情報かどうか注意する
- 専門用語はわかりやすく説明する

日本語で簡潔に応答してください。"""

    return create_react_agent(llm, tools, prompt=system_prompt)


def main():
    """メインループ"""
    print("=" * 50)
    print("デジタルグリッド カスタマーサポート")
    print("終了: quit / exit")
    print("=" * 50)

    agent = create_agent()
    messages = []

    while True:
        try:
            user_input = input("\nYou: ").strip()

            if not user_input:
                continue

            if user_input.lower() in ["quit", "exit", "q"]:
                print("ご利用ありがとうございました。")
                break

            messages.append(HumanMessage(content=user_input))
            result = agent.invoke({"messages": messages})

            ai_message = result["messages"][-1]
            print(f"\nAssistant: {ai_message.content}")

            messages.append(ai_message)

        except KeyboardInterrupt:
            print("\n\n終了します。")
            break
        except Exception as e:
            print(f"\nエラー: {e}")


if __name__ == "__main__":
    main()
