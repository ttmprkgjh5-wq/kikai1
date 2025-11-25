import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import wandb  # ← Weights & Biases をインポート

# ──────────── 設定 ────────────
# ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓ wandb 関連コメント ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓
# wandb.init():
#   ・プロジェクト（ダッシュボード上のひとまとまり）を指定
#   ・実行(run) の名前を付けて区別
#   ・config 引数にハイパーパラメータを渡すと、ダッシュボード上で自動で管理される
#   ・戻り値として “Run” オブジェクトが作られる（ここでは省略しているが wandb.run で参照可）
wandb.init(
    project="mnist-sample",   # プロジェクト名
    name="sample1",           # 実行（ラン）名
    config={                  # ハイパーパラメータなどを記録
        "epochs": 5,
        "batch_size": 64,
        "learning_rate": 1e-2,
    },
)
# wandb.config は init() に渡した config を保持する Dict-Like オブジェクト
config = wandb.config
# ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑ wandb 関連コメント ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑

# ──────────── データ準備 ────────────
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
)

train_ds = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
test_ds = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=config.batch_size, shuffle=False)

# ──────────── モデル定義 ────────────
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleCNN().to(device)

# ──────────── 最適化・損失関数 ────────────
optimizer = optim.SGD(model.parameters(), lr=config.learning_rate)
criterion = nn.CrossEntropyLoss()

# ──────────── 学習ループ ────────────
for epoch in range(1, config.epochs + 1):
    model.train()
    total_loss = 0
    correct = 0
    for batch_idx, (data, target) in enumerate(train_loader, 1):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()

    train_loss = total_loss / len(train_loader)
    train_acc = correct / len(train_loader.dataset)

    # ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓ wandb 関連コメント ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓
    # wandb.log():
    #   ・dict形式で指標を渡すとダッシュボードにリアルタイムで可視化
    #   ・step を指定しなければ内部で自動でインクリメント
    #   ・ここでは epoch ごとにログ
    wandb.log(
        {
            "epoch": epoch,           # 何エポック目か
            "train_loss": train_loss, # 1 エポックの平均損失
            "train_accuracy": train_acc,  # 精度
        }
    )
    # ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑ wandb 関連コメント ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑

    # ───── テスト評価 ─────
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()

    test_loss /= len(test_loader)
    test_acc = correct / len(test_loader.dataset)

    # ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓ wandb 関連コメント ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓
    # テスト結果も同様にログ
    wandb.log(
        {
            "test_loss": test_loss,
            "test_accuracy": test_acc,
        }
    )
    # ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑ wandb 関連コメント ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑

    print(
        f"Epoch {epoch}: "
        f"Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f} | "
        f"Test Loss={test_loss:.4f}, Test Acc={test_acc:.4f}"
    )

# ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓ wandb 関連コメント ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓
# wandb.finish():
#   ・ログのフラッシュと Run の終了処理を行う
#   ・スクリプトの最後に呼ぶのが推奨
wandb.finish()
# ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑ wandb 関連コメント ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑

