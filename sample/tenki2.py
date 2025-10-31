ALL_ITEMS_RULES = {
  "ダウンジャケット": { "max": 5, "min": None },
  "厚手コート": { "max": 5, "min": None },
  "厚手ニット": { "max": 10, "min": None },
  "マフラー": { "max": 10, "min": None },
  "手袋": { "max": 10, "min": None },
  "ニット帽": { "max": 10, "min": None },
  "ヒートテック": { "max": 15, "min": None },
  "コート": { "max": 15, "min": 5 },
  "薄手ダウン": { "max": 15, "min": 5 },
  "セーター": { "max": 15, "min": 5 },
  "タイツ": { "max": 15, "min": None },
  "レギンス": { "max": 20, "min": 10 },
  "ブーツ": { "max": 15, "min": None },
  "薄手ニット": { "max": 20, "min": 10 },
  "パーカー": { "max": 20, "min": 10 },
  "スウェット": { "max": 20, "min": 10 },
  "ジャケット": { "max": 20, "min": 10 },
  "薄手ジャケット": { "max": 25, "min": 15 },
  "カーディガン": { "max": 25, "min": 10 },
  "長袖Tシャツ": { "max": 25, "min": 10 },
  "長袖シャツ": { "max": 25, "min": 10 },
  "ストール": { "max": 20, "min": 10 },
  "ベスト": { "max": 25, "min": 15 },
  "トレーナー": { "max": 20, "min": 10 },
  "ジーンズ": { "max": 25, "min": None },
  "チノパン": { "max": 30, "min": 10 },
  "ロングパンツ": { "max": 30, "min": 10 },
  "スカート": { "max": 30, "min": 15 },
  "ロングスカート": { "max": 30, "min": 15 },
  "七分袖": { "max": 30, "min": 20 },
  "半袖シャツ": { "max": None, "min": 20 },
  "Tシャツ": { "max": None, "min": 15 }, # min:15はインナーとしての着用も考慮
  "ワイドパンツ": { "max": None, "min": 20 },
  "クロップドパンツ": { "max": None, "min": 20 },
  "薄手の羽織り": { "max": 30, "min": 20 },
  "ノースリーブ": { "max": None, "min": 25 },
  "ショートパンツ": { "max": None, "min": 25 },
  "ショートスカート": { "max": None, "min": 25 },
  "リネン素材": { "max": None, "min": 25 },
  "サンダル": { "max": None, "min": 25 },
  "タンクトップ": { "max": None, "min": 28 },
  "UVカットカーディガン": { "max": None, "min": 25 },
  "帽子": { "max": None, "min": 25 },
  "キャップ": { "max": None, "min": 20 },
  "麦わら帽子": { "max": None, "min": 25 },
  "サングラス": { "max": None, "min": 20 },
  "スニーカー": { "max": None, "min": None }, # min, max ともに None = 常にOK
  "パンプス": { "max": 30, "min": 10 },
  "ライトアウター": { "max": 20, "min": 15 },
  "ブルゾン": { "max": 15, "min": 5 },
  "レインコート": { "max": 25, "min": 10 }
}

# --- 2. 自分の持っている服（クローゼット）の定義 ---
my_closet = [
    {'name': '黒のダウンジャケット', 'type': 'ダウンジャケット'},
    {'name': 'グレーのセーター', 'type': 'セーター'},
    {'name': 'ベージュのコート', 'type': 'コート'},
    {'name': 'お気に入りのパーカー', 'type': 'パーカー'},
    {'name': '白のTシャツ', 'type': 'Tシャツ'},
    {'name': 'ブルージーンズ', 'type': 'ジーンズ'},
    {'name': '黒のスキニーパンツ', 'type': 'ロングパンツ'},
    {'name': '白のスニーカー', 'type': 'スニーカー'},
    {'name': '赤のマフラー', 'type': 'マフラー'},
    {'name': 'デニムのショートパンツ', 'type': 'ショートパンツ'},
    {'name': '麦わら帽子', 'type': '麦わら帽子'}
]

# --- 3. ご要望の関数 ---

def get_allowed_clothing_types(temperature, rules):
    print(f"\n--- 関数: 気温{temperature}度に合う服の「種類」を検索 ---")
    
    allowed_types = []
    
    # ルール辞書の全てのアイテムをチェック
    for item_type, rule in rules.items():
        min_temp = rule.get('min')
        max_temp = rule.get('max')
        
        # --- `None` を考慮した気温チェック ---
        
        # 1. min (下限) のチェック
        is_min_ok = False
        if min_temp is None:
            is_min_ok = True  # 下限なし
        elif temperature >= min_temp:
            is_min_ok = True
            
        # 2. max (上限) のチェック
        is_max_ok = False
        if max_temp is None:
            is_max_ok = True  # 上限なし
        elif temperature <= max_temp:
            is_max_ok = True
            
        # 3. 両方の条件を満たした場合、リストに追加
        if is_min_ok and is_max_ok:
            allowed_types.append(item_type)
            
    print(f"  -> 許可された服のタイプ (全{len(allowed_types)}種)")
    # print(allowed_types) # デバッグ用に全種類見たい場合はコメントを外す
    return allowed_types

def filter_my_closet(allowed_types, closet):
    print("\n--- 関数: 自分の服と「許可された種類」を照合 ---")
    
    # 検索を高速化するため、種類のリストを「セット」に変換
    allowed_types_set = set(allowed_types)
    
    wearable_clothes = []
    # クローゼットの中身を1つずつチェック
    for item in closet:
        # 持っている服の「種類」が、許可された種類のセットに含まれているか？
        if item['type'] in allowed_types_set:
            wearable_clothes.append(item)
            print(f"  [OK] {item['name']} (タイプ: {item['type']})")
        else:
            print(f"  [NG] {item['name']} (タイプ: {item['type']}) - 気温に合いません")

    return wearable_clothes

# --- 4. メインの実行部分 ---
if __name__ == "__main__":
    
    current_temperature = 8 
    print(f"*** 現在の気温: {current_temperature}度 ***")
    
    # 1. 関数① を実行
    allowed_types_1 = get_allowed_clothing_types(current_temperature, ALL_ITEMS_RULES)
    
    # 2. 関数② を実行
    wearable_clothes_1 = filter_my_closet(allowed_types_1, my_closet)
    
    # 3. 最終結果の表示
    print("\n--- 最終結果 ---")
    print(f"気温{current_temperature}度で着られるあなたの服:")
    for item in wearable_clothes_1:
        print(f" - {item['name']} ({item['type']})")

    
    current_temperature = 28 
    print(f"*** 現在の気温: {current_temperature}度 ***")
    
    # 1. 関数① を実行
    allowed_types_2 = get_allowed_clothing_types(current_temperature, ALL_ITEMS_RULES)
    
    # 2. 関数② を実行
    wearable_clothes_2 = filter_my_closet(allowed_types_2, my_closet)
    
    # 3. 最終結果の表示
    print("\n--- 最終結果 ---")
    print(f"気温{current_temperature}度で着られるあなたの服:")
    for item in wearable_clothes_2:
        print(f" - {item['name']} ({item['type']})")