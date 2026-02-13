"""
高速チャットボット - 5秒以内の応答を目標
"""

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent

from src.config import settings
from src.utils import create_cached_system_message


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
    from src.rag.llamaindex_hybrid import search_with_context
    return search_with_context(query, k=5)


@tool
def search_customer(
    name: str = "",
    power_area: str = "",
    voltage_type: str = "",
) -> str:
    """顧客（需要地点）情報を検索します。

    検索条件:
    - name: 顧客名・地点名（部分一致）
    - power_area: 電力エリア（Tokyo, Kansai, Chubu, Kyushu, Hokkaido等）
    - voltage_type: 電圧区分（LowVoltage=低圧, HighVoltage=高圧, ExtraHighVoltage=特別高圧）

    Args:
        name: 顧客名（部分一致検索）
        power_area: 電力エリア（英語: Tokyo, Kansai等）
        voltage_type: 電圧区分（LowVoltage/HighVoltage/ExtraHighVoltage）

    Returns:
        検索結果の顧客情報一覧
    """
    from src.customer_data import search_customers, format_customer_info

    results = search_customers(
        name=name if name else None,
        power_area=power_area if power_area else None,
        voltage_type=voltage_type if voltage_type else None,
        active_only=True,
    )

    if not results:
        return "該当する顧客が見つかりませんでした。"

    output = f"【検索結果: {len(results)}件】\n\n"
    for customer in results[:10]:  # 最大10件
        output += format_customer_info(customer) + "\n\n"

    if len(results) > 10:
        output += f"※他 {len(results) - 10}件の顧客がヒットしています。条件を絞り込んでください。"

    return output


@tool
def get_customer_by_id(customer_id: int) -> str:
    """顧客IDで顧客情報を取得します。

    Args:
        customer_id: 顧客ID（数値）

    Returns:
        顧客の詳細情報
    """
    from src.customer_data import get_customer_by_id as get_customer, format_customer_info

    customer = get_customer(customer_id)
    if customer is None:
        return f"顧客ID {customer_id} は見つかりませんでした。"

    return format_customer_info(customer)


def create_agent():
    """エージェントを作成"""
    llm = ChatOpenAI(
        model=settings.OPENAI_MODEL,
        temperature=0.3,
    )

    tools = [search_manual, search_customer, get_customer_by_id]

    system_prompt = """あなたはデジタルグリッド株式会社の顧客ポータル「DGM（Digital Grid Manager）」内に設置されたカスタマーサポートAIです。

【重要な前提】
- ユーザーは今まさにDGMの画面を操作しています
- システムマニュアルは、このDGMの操作説明書です
- 画面操作やボタンに関する質問は、システムマニュアルの情報を優先してください

【利用可能なツール】
1. search_manual: マニュアル・約款の検索
   - システムマニュアル: DGMの画面操作、メニュー、ボタンの説明
   - 電気需給約款: 契約条件、料金計算、支払い
   - 託送供給等約款: 各電力会社エリアの技術基準・料金
   - DGP利用規約: プラットフォーム利用条件、データ取扱い

2. search_customer: 顧客（需要地点）情報の検索
   - 顧客名、電力エリア、電圧区分で検索可能
   - 電力エリア: Tokyo, Kansai, Chubu, Kyushu, Hokkaido, Tohoku, Hokuriku, Chugoku, Shikoku, Okinawa
   - 電圧区分: LowVoltage（低圧）, HighVoltage（高圧）, ExtraHighVoltage（特別高圧）

3. get_customer_by_id: 顧客IDで詳細情報を取得

【回答のルール】
- 顧客情報に関する質問はsearch_customerまたはget_customer_by_idを使用
- マニュアル・約款に関する質問はsearch_manualを使用
- ドキュメントにない情報は推測せず「カスタマーサポートにお問い合わせください」と案内
- 専門用語はわかりやすく説明する

日本語で簡潔に応答してください。"""

    # Prompt Caching を適用（Anthropic: cache_control設定、OpenAI: 自動キャッシュ）
    cached_prompt = create_cached_system_message(system_prompt)

    return create_react_agent(llm, tools, prompt=cached_prompt)


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
