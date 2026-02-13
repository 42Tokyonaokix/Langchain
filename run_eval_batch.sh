#!/bin/bash
# 評価スクリプトを複数回実行するバッチスクリプト
#
# 使用例:
#   ./run_eval_batch.sh 5                           # 5回実行
#   ./run_eval_batch.sh 3 --test=data/test_cases2.csv  # 別のテストファイルで3回実行
#   ./run_eval_batch.sh 5 --max=10                  # 最大10件で5回実行

cd "$(dirname "$0")"

# 引数チェック
if [ -z "$1" ]; then
    echo "使用方法: $0 <回数> [オプション...]"
    echo ""
    echo "例:"
    echo "  $0 5                              # 5回実行"
    echo "  $0 3 --test=data/test_cases2.csv  # 別のテストファイルで3回実行"
    echo "  $0 5 --max=10                     # 最大10件で5回実行"
    exit 1
fi

COUNT=$1
shift  # 最初の引数（回数）を除去し、残りをオプションとして渡す

echo "=========================================="
echo "評価バッチ実行: ${COUNT}回"
echo "オプション: $@"
echo "=========================================="

for i in $(seq 1 $COUNT); do
    echo ""
    echo ">>> 実行 ${i}/${COUNT} <<<"
    echo ""
    python src/evaluation/local.py "$@"

    # 最後の実行以外は少し待機（API制限対策）
    if [ $i -lt $COUNT ]; then
        echo ""
        echo "次の実行まで5秒待機..."
        sleep 5
    fi
done

echo ""
echo "=========================================="
echo "全${COUNT}回の実行が完了しました"
echo "=========================================="
