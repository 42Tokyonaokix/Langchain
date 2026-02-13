"""
電力約款用RAGデータセット作成スクリプト
LangSmithにデータセットをアップロードします

test_cases.csv からQ&Aデータを読み込みます
"""
import sys
import csv
from pathlib import Path

# プロジェクトルートをパスに追加
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from langsmith import Client

from src.config import settings

# APIキー確認
if not settings.LANGCHAIN_API_KEY or settings.LANGCHAIN_API_KEY == "your-langsmith-api-key":
    print("エラー: LANGCHAIN_API_KEY が設定されていません")
    print(".env ファイルに有効なLangSmith APIキーを設定してください")
    print("APIキーは https://smith.langchain.com/settings で取得できます")
    exit(1)

client = Client()

# パス設定
DEFAULT_CSV_PATH = PROJECT_ROOT / "data" / "test_cases.csv"


def load_qa_from_csv(csv_path: Path) -> list[dict]:
    """CSVファイルからQ&Aデータを読み込む"""
    examples = []

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # 新形式（質問/回答）と旧形式（question/answer）の両方に対応
            question = row.get("質問", row.get("question", "")).strip()
            answer = row.get("回答", row.get("answer", "")).strip()

            # 空行をスキップ
            if not question or not answer:
                continue

            examples.append({
                "inputs": {"question": question},
                "outputs": {"answer": answer},
            })

    return examples


def get_unique_dataset_name(base_name: str) -> str:
    """重複しないデータセット名を取得（番号を増やしていく）"""
    # まずベース名で試す
    try:
        client.read_dataset(dataset_name=base_name)
    except Exception:
        return base_name  # 存在しないのでそのまま使用

    # 番号付きで試す
    n = 2
    while True:
        candidate = f"{base_name}_{n}"
        try:
            client.read_dataset(dataset_name=candidate)
            n += 1
        except Exception:
            return candidate


def main():
    # コマンドライン引数の解析
    csv_path = DEFAULT_CSV_PATH
    dataset_name = None

    for arg in sys.argv[1:]:
        if arg.startswith("--test="):
            csv_path = Path(arg.split("=")[1])
        elif arg.startswith("--name="):
            dataset_name = arg.split("=")[1]

    # データセット名をファイル名から自動生成（指定がない場合）
    if dataset_name is None:
        base_name = f"electricity-yakkan-qa-{csv_path.stem}"
    else:
        base_name = dataset_name

    # CSVからQ&Aデータを読み込み
    if not csv_path.exists():
        print(f"エラー: CSVファイルが見つかりません: {csv_path}")
        exit(1)

    examples = load_qa_from_csv(csv_path)

    if not examples:
        print("エラー: Q&Aデータが読み込めませんでした")
        exit(1)

    print(f"CSVから {len(examples)} 件のQ&Aデータを読み込みました")
    print(f"テストケース: {csv_path}")

    # 重複しないデータセット名を取得
    dataset_name = get_unique_dataset_name(base_name)
    if dataset_name != base_name:
        print(f"データセット名 '{base_name}' は既に存在するため '{dataset_name}' を使用します")

    # データセット作成
    dataset = client.create_dataset(
        dataset_name=dataset_name,
        description=f"電力約款に関するQ&Aデータセット（RAG評価用）- {csv_path.name}から生成"
    )

    # サンプル登録
    client.create_examples(dataset_id=dataset.id, examples=examples)

    print(f"データセット '{dataset_name}' を作成しました")
    print(f"登録件数: {len(examples)}件")
    print(f"データセットID: {dataset.id}")


if __name__ == "__main__":
    main()
