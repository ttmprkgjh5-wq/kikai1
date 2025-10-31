ALL_ITEMS_RULES = {
    'ダウン': {'min': -10, 'max': 5}, '厚手ニット': {'min': -10, 'max': 5}, 'マフラー': {'min': -10, 'max': 5},
    'コート': {'min': 5, 'max': 10}, 'セーター': {'min': 5, 'max': 10}, '薄手ダウン': {'min': 5, 'max': 10},
    'パーカー': {'min': 10, 'max': 15}, 'スウェット': {'min': 10, 'max': 15}, '薄手ジャケット': {'min': 10, 'max': 15},
    'トレーナー': {'min': 15, 'max': 20}, 'シャツ': {'min': 15, 'max': 20}, '長袖Tシャツ': {'min': 15, 'max': 20},
    '半袖Tシャツ': {'min': 20, 'max': 25}, '七分袖': {'min': 20, 'max': 25},
    'ノースリーブ': {'min': 25, 'max': 40}, 'リネン素材': {'min': 25, 'max': 40}, 'ショートパンツ': {'min': 25, 'max': 40},
    'デニム': {'min': 5, 'max': 25}, 'チノパン': {'min': 10, 'max': 30}, 'スカート': {'min': 15, 'max': 35}
}

my_closet = [
    {'name': '黒のダウンジャケット', 'type': 'ダウン'},
    {'name': 'グレーのウールコート', 'type': 'コート'},
    {'name': 'お気に入りのパーカー', 'type': 'パーカー'},
    {'name': '白の長袖Tシャツ', 'type': '長袖Tシャツ'},
    {'name': 'チェック柄のシャツ', 'type': 'シャツ'},
    {'name': '青の半袖Tシャツ', 'type': '半袖Tシャツ'},
    {'name': '麻のノースリーブ', 'type': 'ノースリーブ'},
    {'name': 'いつものジーパン', 'type': 'デニム'},
    {'name': 'ベージュのチノパン', 'type': 'チノパン'},
    {'name': '黒のショートパンツ', 'type': 'ショートパンツ'}
]

def filter_closet_by_temp(temperature, closet, rules):
    """
    気温とクローゼット、ルールを入力とし、
    着られる服のリストを返す関数
    """
    print(f"\n--- 気温{temperature}度でクローゼットを検索 ---")
    
    wearable_clothes = []
    # クローゼットの中身を1つずつチェック
    for item in closet:
        item_type = item['type']
        
        # ルール辞書にその服の種類（キー）が存在するか確認
        if item_type in rules:
            # ルールを取得
            rule = rules[item_type] 
            
            # 気温がルールの範囲内かチェック
            if rule['min'] <= temperature <= rule['max']:
                wearable_clothes.append(item)
                print(f"  [OK] {item['name']} (タイプ: {item_type})")
            else:
                print(f"  [NG] {item['name']} (タイプ: {item_type}) - 気温が範囲外 ({rule['min']}～{rule['max']}℃)")
        else:
            # ルール辞書に登録されていない服
            print(f"  [?] {item['name']} (タイプ: {item_type}) - ルールが未定義です")
            
    return wearable_clothes

if __name__ == "__main__":
    
    # --- テスト実行 ---
    current_temperature = 14 # (10〜15℃の範囲をテスト)
    
    print(f"*** 現在の気温: {current_temperature}度 ***")
    
    # 1. 関数を実行
    wearable_clothes = filter_closet_by_temp(current_temperature, my_closet, ALL_ITEMS_RULES)
    
    # 2. 最終結果の表示
    print("\n--- 最終結果 ---")
    print(f"気温{current_temperature}度で着られるあなたの服:")
    if wearable_clothes:
        for item in wearable_clothes:
            print(f" - {item['name']} ({item['type']})")
    else:
        print("着られる服がありません。")