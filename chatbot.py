"""LangChain Agent Chatbot"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage

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
def search_web(query: str) -> str:
    """ウェブ検索をシミュレートします（実際の検索機能は未実装）。

    Args:
        query: 検索クエリ
    """
    # 実際の実装では、SerpAPI や Tavily などの検索APIを使用
    return f"「{query}」の検索結果: この機能は現在シミュレーションモードです。実際の検索APIを統合してください。"


def create_agent():
    """エージェントを作成"""
    model_name = os.getenv("OPENAI_MODEL", "gpt-4o")

    llm = ChatOpenAI(
        model=model_name,
        temperature=0.7,
    )

    tools = [calculator, get_current_time, search_web]

    prompt = ChatPromptTemplate.from_messages([
        ("system", """あなたは親切で有能なAIアシスタントです。
ユーザーの質問に対して、必要に応じてツールを使用して回答してください。
日本語で応答してください。"""),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    agent = create_tool_calling_agent(llm, tools, prompt)

    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,  # ツール呼び出しの詳細を表示
        handle_parsing_errors=True,
    )


def main():
    """メインループ"""
    print("=" * 50)
    print("LangChain Agent Chatbot")
    print("終了するには 'quit' または 'exit' と入力してください")
    print("=" * 50)

    agent_executor = create_agent()
    chat_history = []

    while True:
        try:
            user_input = input("\nYou: ").strip()

            if not user_input:
                continue

            if user_input.lower() in ["quit", "exit", "q"]:
                print("さようなら！")
                break

            # エージェントを実行
            result = agent_executor.invoke({
                "input": user_input,
                "chat_history": chat_history,
            })

            response = result["output"]
            print(f"\nAssistant: {response}")

            # 会話履歴を更新
            chat_history.append(HumanMessage(content=user_input))
            chat_history.append(AIMessage(content=response))

        except KeyboardInterrupt:
            print("\n\n中断されました。さようなら！")
            break
        except Exception as e:
            print(f"\nエラーが発生しました: {e}")


if __name__ == "__main__":
    main()
