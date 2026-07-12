"""ヒアリング11項目の内部フィールド名と日本語項目名の対応（共通定義）

workflow（ヒアリング完了ガード）と case_memory（事例プロファイル生成）で
共有する。項目を追加・変更する場合はここだけを修正する。
"""

FIELD_JA = {
    "equipment_type":     "設備の種類",
    "installation_place": "設置場所",
    "operation_purpose":  "用途・目的",
    "scheduled_date":     "稼働開始予定日",
    "chemicals":          "薬品・溶剤・ガス・燃料",
    "fire_exhaust":       "火気・熱源・排気・粉じん",
    "wastewater":         "排水・廃液・廃棄物",
    "noise_vibration":    "騒音・振動",
    "radiation":          "放射線・X線",
    "construction":       "建屋改修・電気工事・配管工事",
    "additional_info":    "その他の情報",
}
