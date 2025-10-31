import cv2
import numpy as np

# ====== 加重平均関数 ======
def average(values, weights):
    return np.sum(values * weights) / np.sum(weights) 

def circular_average(h_values, weights):
    h_deg = h_values * 2  # OpenCVではHは0–179なので×2して0–358°に
    h_rad = np.deg2rad(h_deg)

    x_mean = np.sum(np.cos(h_rad) * weights) 
    y_mean = np.sum(np.sin(h_rad) * weights) 

    angle = np.rad2deg(np.arctan2(y_mean, x_mean)) % 360
    return angle / 2  # OpenCVスケールに戻す（0-179）


# ====== メイン処理 ======
# 画像の読み込み
image = cv2.imread('project_photo/test2.jpg')
if image is None:
    raise FileNotFoundError("画像が見つかりません。")

# BGRからHSVへの変換
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
H, S, V = cv2.split(hsv_image)

# 画像サイズ
h, w, _ = hsv_image.shape
y, x = np.ogrid[:h, :w]

# 中央を基準にガウス分布で重みを作る
cx, cy = w / 2, h / 2
sigma_x, sigma_y = w / 4, h / 4  # 中央の重み範囲
weight = np.exp(-(((x - cx) ** 2) / (2 * sigma_x ** 2) + ((y - cy) ** 2) / (2 * sigma_y ** 2)))

# 重みを正規化
if np.sum(weight) == 0:
    raise ValueError("有効なピクセルがありません。")
weight /= np.sum(weight)

# ====== 各チャンネルの加重平均 ======
H_mean = average(H, weight)
S_mean = average(S, weight)
V_mean = average(V, weight)

# 出力
rep_color_hsv = (H_mean, S_mean, V_mean)
rep_color_bgr = cv2.cvtColor(np.uint8([[[H_mean, S_mean, V_mean]]]), cv2.COLOR_HSV2BGR)[0][0]

print("代表色（HSV）：", rep_color_hsv)
print("代表色（BGR）：", rep_color_bgr)


# ====== 可視化 ======
# 代表色の画像を作成
rep_color_image = np.full((100, 100, 3), rep_color_bgr, dtype=np.uint8)
cv2.imshow('Representative Color', rep_color_image)
cv2.waitKey(0)
cv2.destroyAllWindows()