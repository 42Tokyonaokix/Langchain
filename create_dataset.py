"""
電力約款用RAGデータセット作成スクリプト
LangSmithにデータセットをアップロードします
"""
import os
from dotenv import load_dotenv
from langsmith import Client

# .envファイルを読み込み
load_dotenv()

# APIキー確認
api_key = os.getenv("LANGCHAIN_API_KEY")
if not api_key or api_key == "your-langsmith-api-key":
    print("エラー: LANGCHAIN_API_KEY が設定されていません")
    print(".env ファイルに有効なLangSmith APIキーを設定してください")
    print("APIキーは https://smith.langchain.com/settings で取得できます")
    exit(1)

client = Client()

# データセット名
DATASET_NAME = "electricity-yakkan-qa"

# 約款に基づくQ&Aデータ
examples = [
    # ===== 契約期間 =====
    {
        "inputs": {"question": "契約期間はどのくらいですか？"},
        "outputs": {"answer": "基本契約の契約期間は、契約者と当社との協議が整った日から1年間です。契約期間満了に先だって契約内容に変更がない場合は、契約期間満了後も1年ごとに同一条件で継続されます。"},
    },
    {
        "inputs": {"question": "契約は自動更新されますか？"},
        "outputs": {"answer": "はい、契約期間満了に先だって契約内容に変更がない場合は、契約期間満了後も1年ごとに同一条件で自動的に継続されます。"},
    },

    # ===== 届出・届出事項 =====
    {
        "inputs": {"question": "届出が必要な場合はどんな時ですか？"},
        "outputs": {"answer": "契約者は、需要場所、契約者の名義、使用状況などに変更があった場合は、すみやかに届け出る必要があります。届出がない場合、供給停止などの措置が取られることがあります。"},
    },
    {
        "inputs": {"question": "名義変更の届出は必要ですか？"},
        "outputs": {"answer": "はい、契約者の名義に変更があった場合は、すみやかに届け出る必要があります。届出がない場合、契約の継続に支障が生じる可能性があります。"},
    },

    # ===== 供給停止 =====
    {
        "inputs": {"question": "電気の供給が停止される場合はどんな時ですか？"},
        "outputs": {"answer": "料金の支払いが遅延した場合、届出事項に虚偽があった場合、契約に違反した場合などに、供給を停止することがあります。また、保安上の理由や系統の制約などにより、やむを得ず供給を停止することもあります。"},
    },
    {
        "inputs": {"question": "料金を払わないとどうなりますか？"},
        "outputs": {"answer": "料金のお支払いが遅延した場合、供給を停止することがあります。支払期日を過ぎた場合は、延滞利息が発生する場合もあります。"},
    },

    # ===== 解約 =====
    {
        "inputs": {"question": "解約したい場合はどうすればいいですか？"},
        "outputs": {"answer": "解約をご希望の場合は、事前に届け出ていただく必要があります。解約希望日の前までにお申し出ください。"},
    },
    {
        "inputs": {"question": "解約に違約金はかかりますか？"},
        "outputs": {"answer": "約款上、契約期間中の解約に関する違約金の規定は契約内容によって異なります。詳細は契約書またはお客様センターにお問い合わせください。"},
    },

    # ===== 料金 =====
    {
        "inputs": {"question": "料金の支払い期日はいつですか？"},
        "outputs": {"answer": "料金のお支払いは、検針日の翌日から起算して30日目までにお支払いいただきます。支払期日を過ぎた場合は、延滞利息が発生することがあります。"},
    },
    {
        "inputs": {"question": "延滞利息はどのくらいですか？"},
        "outputs": {"answer": "支払期日を過ぎてもお支払いがない場合、延滞利息として年率約10％（1日あたり約0.03％）が発生する場合があります。詳細は約款をご確認ください。"},
    },

    # ===== 接続供給 =====
    {
        "inputs": {"question": "接続供給契約とは何ですか？"},
        "outputs": {"answer": "接続供給契約とは、小売電気事業者等が需要家に電気を供給するために、一般送配電事業者の送配電ネットワークを利用する契約です。"},
    },
    {
        "inputs": {"question": "振替供給とは何ですか？"},
        "outputs": {"answer": "振替供給とは、発電事業者等が発電した電気を、送配電ネットワークを介して他のエリアや需要家に届けるためのサービスです。"},
    },

    # ===== 保安・工事 =====
    {
        "inputs": {"question": "計器の設置や交換は誰がやりますか？"},
        "outputs": {"answer": "計量器（メーター）の設置、交換、検査等は、一般送配電事業者が行います。計量器に異常を発見した場合は、すみやかにご連絡ください。"},
    },
    {
        "inputs": {"question": "停電の際はどうすればいいですか？"},
        "outputs": {"answer": "停電が発生した場合は、まず周辺地域の状況を確認してください。地域全体の停電の場合は送配電事業者が対応します。お客様の設備のみの場合は、ブレーカーを確認し、それでも復旧しない場合はお客様センターにご連絡ください。"},
    },
]


def main():
    # データセット作成
    dataset = client.create_dataset(
        dataset_name=DATASET_NAME,
        description="電力約款に関するQ&Aデータセット（RAG評価用）"
    )

    # サンプル登録
    client.create_examples(dataset_id=dataset.id, examples=examples)

    print(f"データセット '{DATASET_NAME}' を作成しました")
    print(f"登録件数: {len(examples)}件")
    print(f"データセットID: {dataset.id}")


if __name__ == "__main__":
    main()
