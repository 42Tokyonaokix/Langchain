"""mockデータの動作確認"""

from mock_data import (
    get_all_customers,
    get_customer_by_id,
    search_customers,
    format_customer_info,
)


def main():
    print("=" * 50)
    print("顧客mockデータ動作確認")
    print("=" * 50)

    # 全顧客取得
    print("\n【全顧客一覧】")
    customers = get_all_customers()
    print(f"顧客数: {len(customers)}件")

    # ID検索
    print("\n【ID検索: 1001】")
    customer = get_customer_by_id(1001)
    if customer:
        print(format_customer_info(customer))

    # エリア検索
    print("\n【東京エリアの顧客】")
    tokyo_customers = search_customers(power_area="Tokyo")
    for c in tokyo_customers:
        print(f"  - {c['endpoint_name']} ({c['voltage_type']})")

    # 電圧区分検索
    print("\n【特別高圧の顧客】")
    extra_high = search_customers(voltage_type="ExtraHighVoltage")
    for c in extra_high:
        print(f"  - {c['endpoint_name']} ({c['power_area']})")

    # 名前検索
    print("\n【名前検索: '工場'】")
    factories = search_customers(name="工場")
    for c in factories:
        print(f"  - {c['endpoint_name']}")

    # 契約終了含む検索
    print("\n【契約終了含む全件】")
    all_including_ended = search_customers(active_only=False)
    print(f"全件: {len(all_including_ended)}件")
    ended = [c for c in all_including_ended if c["contract_finish_date"]]
    print(f"契約終了: {len(ended)}件")


if __name__ == "__main__":
    main()
