"""
電力約款用RAGデータセット作成スクリプト
LangSmithにデータセットをアップロードします

documents/qa_list.csv からQ&Aデータを読み込みます
"""
import os
import csv
from pathlib import Path
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

# CSVファイルパス
PROJECT_ROOT = Path(__file__).parent.parent.parent
CSV_PATH = PROJECT_ROOT / "documents" / "qa_list.csv"


def load_qa_from_csv(csv_path: Path) -> list[dict]:
    """CSVファイルからQ&Aデータを読み込む"""
    examples = []

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            question = row.get("question", "").strip()
            answer = row.get("answer", "").strip()

            # 空行をスキップ
            if not question or not answer:
                continue

            examples.append({
                "inputs": {"question": question},
                "outputs": {"answer": answer},
            })

    return examples


def main():
    # CSVからQ&Aデータを読み込み
    if not CSV_PATH.exists():
        print(f"エラー: CSVファイルが見つかりません: {CSV_PATH}")
        exit(1)

    examples = load_qa_from_csv(CSV_PATH)

    if not examples:
        print("エラー: Q&Aデータが読み込めませんでした")
        exit(1)

    print(f"CSVから {len(examples)} 件のQ&Aデータを読み込みました")

    # データセット作成
    dataset = client.create_dataset(
        dataset_name=DATASET_NAME,
        description="電力約款に関するQ&Aデータセット（RAG評価用）- documents/qa_list.csvから生成"
    )

    # サンプル登録
    client.create_examples(dataset_id=dataset.id, examples=examples)

    print(f"データセット '{DATASET_NAME}' を作成しました")
    print(f"登録件数: {len(examples)}件")
    print(f"データセットID: {dataset.id}")


if __name__ == "__main__":
    main()
