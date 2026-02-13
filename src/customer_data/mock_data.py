"""
dgm-backend互換の顧客（Endpoint）mockデータ

データ構造はdgm-backendのRetailAgentAppEndpointResponseModelに準拠
"""

from datetime import date
from typing import TypedDict, Literal


# dgm-backendのEnum定義に合わせた型
VoltageType = Literal["LowVoltage", "HighVoltage", "ExtraHighVoltage"]
PowerArea = Literal[
    "Hokkaido", "Tohoku", "Tokyo", "Chubu", "Hokuriku",
    "Kansai", "Chugoku", "Shikoku", "Kyushu", "Okinawa"
]


class CustomerEndpoint(TypedDict):
    """顧客エンドポイント（需要地点）"""
    id: int
    endpoint_name: str  # 地点名/顧客名
    voltage_type: VoltageType  # 電圧区分: 低圧/高圧/特別高圧
    power_area: PowerArea  # 電力エリア
    demand_point: str | None  # 供給地点特定番号（22桁）
    contract_start_date: str | None  # 契約開始日 (YYYY-MM-DD)
    contract_finish_date: str | None  # 契約終了日 (YYYY-MM-DD)


# 電力エリア日本語名マッピング
POWER_AREA_JP = {
    "Hokkaido": "北海道",
    "Tohoku": "東北",
    "Tokyo": "東京",
    "Chubu": "中部",
    "Hokuriku": "北陸",
    "Kansai": "関西",
    "Chugoku": "中国",
    "Shikoku": "四国",
    "Kyushu": "九州",
    "Okinawa": "沖縄",
}

# 電圧区分日本語名マッピング
VOLTAGE_TYPE_JP = {
    "LowVoltage": "低圧",
    "HighVoltage": "高圧",
    "ExtraHighVoltage": "特別高圧",
}


# ==================== Mockデータ ====================
MOCK_CUSTOMERS: list[CustomerEndpoint] = [
    # 東京エリアの顧客
    {
        "id": 1001,
        "endpoint_name": "株式会社サンプル商事 本社ビル",
        "voltage_type": "HighVoltage",
        "power_area": "Tokyo",
        "demand_point": "0300000000000000001001",
        "contract_start_date": "2023-04-01",
        "contract_finish_date": None,
    },
    {
        "id": 1002,
        "endpoint_name": "株式会社サンプル商事 東京支店",
        "voltage_type": "LowVoltage",
        "power_area": "Tokyo",
        "demand_point": "0300000000000000001002",
        "contract_start_date": "2023-06-01",
        "contract_finish_date": None,
    },
    {
        "id": 1003,
        "endpoint_name": "東京データセンター",
        "voltage_type": "ExtraHighVoltage",
        "power_area": "Tokyo",
        "demand_point": "0300000000000000001003",
        "contract_start_date": "2022-01-01",
        "contract_finish_date": None,
    },
    # 関西エリアの顧客
    {
        "id": 2001,
        "endpoint_name": "関西物流センター",
        "voltage_type": "HighVoltage",
        "power_area": "Kansai",
        "demand_point": "0600000000000000002001",
        "contract_start_date": "2023-01-15",
        "contract_finish_date": None,
    },
    {
        "id": 2002,
        "endpoint_name": "大阪工場",
        "voltage_type": "ExtraHighVoltage",
        "power_area": "Kansai",
        "demand_point": "0600000000000000002002",
        "contract_start_date": "2021-07-01",
        "contract_finish_date": None,
    },
    # 中部エリアの顧客
    {
        "id": 3001,
        "endpoint_name": "名古屋オフィス",
        "voltage_type": "HighVoltage",
        "power_area": "Chubu",
        "demand_point": "0400000000000000003001",
        "contract_start_date": "2023-09-01",
        "contract_finish_date": None,
    },
    {
        "id": 3002,
        "endpoint_name": "中部製造工場",
        "voltage_type": "ExtraHighVoltage",
        "power_area": "Chubu",
        "demand_point": "0400000000000000003002",
        "contract_start_date": "2020-04-01",
        "contract_finish_date": None,
    },
    # 九州エリアの顧客
    {
        "id": 4001,
        "endpoint_name": "福岡支社",
        "voltage_type": "HighVoltage",
        "power_area": "Kyushu",
        "demand_point": "0900000000000000004001",
        "contract_start_date": "2024-01-01",
        "contract_finish_date": None,
    },
    # 北海道エリアの顧客
    {
        "id": 5001,
        "endpoint_name": "札幌冷凍倉庫",
        "voltage_type": "HighVoltage",
        "power_area": "Hokkaido",
        "demand_point": "0100000000000000005001",
        "contract_start_date": "2022-11-01",
        "contract_finish_date": None,
    },
    # 契約終了した顧客
    {
        "id": 9001,
        "endpoint_name": "旧・横浜倉庫",
        "voltage_type": "HighVoltage",
        "power_area": "Tokyo",
        "demand_point": "0300000000000000009001",
        "contract_start_date": "2021-01-01",
        "contract_finish_date": "2023-12-31",
    },
]


def get_all_customers() -> list[CustomerEndpoint]:
    """全顧客データを取得"""
    return MOCK_CUSTOMERS


def get_customer_by_id(customer_id: int) -> CustomerEndpoint | None:
    """IDで顧客を検索"""
    for customer in MOCK_CUSTOMERS:
        if customer["id"] == customer_id:
            return customer
    return None


def search_customers(
    name: str | None = None,
    power_area: PowerArea | None = None,
    voltage_type: VoltageType | None = None,
    active_only: bool = True,
) -> list[CustomerEndpoint]:
    """顧客を検索

    Args:
        name: 顧客名（部分一致）
        power_area: 電力エリア
        voltage_type: 電圧区分
        active_only: 契約中のみ（True）/ 全件（False）
    """
    results = []
    for customer in MOCK_CUSTOMERS:
        # 契約終了フィルター
        if active_only and customer["contract_finish_date"] is not None:
            continue
        # 名前フィルター
        if name and name not in customer["endpoint_name"]:
            continue
        # エリアフィルター
        if power_area and customer["power_area"] != power_area:
            continue
        # 電圧区分フィルター
        if voltage_type and customer["voltage_type"] != voltage_type:
            continue
        results.append(customer)
    return results


def format_customer_info(customer: CustomerEndpoint) -> str:
    """顧客情報を日本語でフォーマット"""
    area_jp = POWER_AREA_JP.get(customer["power_area"], customer["power_area"])
    voltage_jp = VOLTAGE_TYPE_JP.get(customer["voltage_type"], customer["voltage_type"])

    status = "契約中"
    if customer["contract_finish_date"]:
        status = f"契約終了 ({customer['contract_finish_date']})"

    return f"""【顧客ID: {customer['id']}】
地点名: {customer['endpoint_name']}
エリア: {area_jp}
電圧区分: {voltage_jp}
供給地点特定番号: {customer['demand_point'] or '未設定'}
契約開始日: {customer['contract_start_date']}
ステータス: {status}"""
