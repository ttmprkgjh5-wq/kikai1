import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms

# GPU・最適化アルゴリズムの設定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
unet = unet.to(device)
optimizer = torch.optim.Adam(unet.parameters(), lr=0.001)

# 損失関数の設定
loss = nn.CrossEntropyLoss()
input = torch.randn(3, 5, requires_grad=True)
target = torch.empty(3, dtype=torch.long).random_(5)
output = loss(input, target)
output.backward()

# 学習ループ