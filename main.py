import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import warnings
import copy
from torch.utils.data import DataLoader, TensorDataset
warnings.filterwarnings("ignore")

import random
import os

# 固定随机种子
seed = 1024
os.environ["PL_GLOBAL_SEED"] = str(seed)
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ================= 1. 数据预处理 (移到循环外部，只做一次) =================
print("Loading and preprocessing data...")
df = pd.read_csv('Helpdesk.csv')
df['CompleteTimestamp'] = pd.to_datetime(df['time:timestamp'], format='%Y/%m/%d %H:%M:%S')

# 全局拟合 LabelEncoder
label_encoder = LabelEncoder()
df['ActivityID_encoded'] = label_encoder.fit_transform(df['concept:name'])
num_classes = len(label_encoder.classes_)
PAD_IDX = num_classes  # 使用类别总数作为 Padding 索引

# 提取所有的轨迹数据
cases = []
for case_id, group in df.groupby('case:concept:name'):
    if len(group) <= 1:
        continue
    events = group.sort_values(by='CompleteTimestamp')
    acts = events['ActivityID_encoded'].tolist()
    times = events['CompleteTimestamp'].tolist()
    cases.append({'acts': acts, 'times': times})

print(f"Total valid cases: {len(cases)}")
print(f"Vocabulary size (num_classes): {num_classes}, PAD_IDX: {PAD_IDX}")

# ================= 2. 模型定义 =================
# 任务一：预测下一个活动的分类模型
class ActivityPredictor(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_hidden, num_out, pad_idx):
        super(ActivityPredictor, self).__init__()
        self.embedding = nn.Embedding(vocab_size + 1, embedding_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(embedding_dim, num_hidden, num_layers=1, batch_first=True)
        self.attention = nn.Sequential(
            nn.Linear(num_hidden, 1),
            nn.Tanh()
        )
        self.fc = nn.Linear(num_hidden, num_out)

    def forward(self, x):
        embeds = self.embedding(x)
        out, _ = self.lstm(embeds)
        
        attn_weights = torch.softmax(self.attention(out), dim=1)
        attended_out = torch.sum(out * attn_weights, dim=1)
        
        return self.fc(attended_out) # CrossEntropyLoss 会自带 Softmax

# 任务二：预测剩余时间的回归模型
class TimePredictor(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_hidden, pad_idx):
        super(TimePredictor, self).__init__()
        self.embedding = nn.Embedding(vocab_size + 1, embedding_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(embedding_dim, num_hidden, num_layers=1, batch_first=True)
        self.attention = nn.Sequential(
            nn.Linear(num_hidden, 1),
            nn.Tanh()
        )
        self.fc = nn.Linear(num_hidden, 1)

    def forward(self, x):
        embeds = self.embedding(x)
        out, _ = self.lstm(embeds)
        
        attn_weights = torch.softmax(self.attention(out), dim=1)
        attended_out = torch.sum(out * attn_weights, dim=1)
        
        return self.fc(attended_out).squeeze(-1)

# ================= 3. 通用训练和评估引擎 =================
def train_and_evaluate(model, train_loader, val_loader, test_loader, criterion, optimizer, device, is_regression=False, patience=50, epochs=2000):
    best_val_loss = float('inf')
    best_model_wts = copy.deepcopy(model.state_dict())
    no_improv = 0

    for epoch in range(epochs):
        model.train()
        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            preds = model(X_b)
            loss = criterion(preds, y_b)
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_b, y_b in val_loader:
                X_b, y_b = X_b.to(device), y_b.to(device)
                preds = model(X_b)
                loss = criterion(preds, y_b)
                val_loss += loss.item()
        val_loss /= len(val_loader)

        # Early stopping logic (保存到内存而非硬盘)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            no_improv = 0
        else:
            no_improv += 1

        if no_improv >= patience:
            break

    # 加载最佳权重并在测试集上评估
    model.load_state_dict(best_model_wts)
    model.eval()
    test_metric = 0.0
    total_samples = 0

    with torch.no_grad():
        for X_b, y_b in test_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            preds = model(X_b)
            
            if is_regression:
                # 回归任务计算 MAE (Mean Absolute Error) 更直观
                test_metric += nn.functional.l1_loss(preds, y_b, reduction='sum').item()
            else:
                # 分类任务计算准确率
                _, predicted = torch.max(preds, 1)
                test_metric += (predicted == y_b).sum().item()
            
            total_samples += y_b.size(0)

    return test_metric / total_samples

# ================= 4. 前缀遍历实验 =================
prefix_lengths = list(range(2, 10))
precisions = []
mae_results = []

batch_size = 64
lr = 0.0001

for i in prefix_lengths:
    X, y1, y2 = [], [], []

    for case in cases:
        acts = case['acts']
        times = case['times']
        n_events = len(acts)

        if n_events <= i:
            # 截取除最后一个外的所有作为前缀，并在右侧填充
            prefix = acts[:-1]
            pad_len = i - len(prefix)
            x_seq = prefix + [PAD_IDX] * pad_len
            label_y1 = acts[-1]
            # 剩余时间：轨迹最后一个时间戳 - 前缀最后一个事件的时间戳
            time_diff_seconds = (times[-1] - times[-2]).total_seconds()
        else:
            x_seq = acts[0:i]
            label_y1 = acts[i]
            # 剩余时间：轨迹最后一个时间戳 - 前缀最后一个事件的时间戳
            time_diff_seconds = (times[-1] - times[i-1]).total_seconds()

        X.append(x_seq)
        y1.append(label_y1)
        # 将秒转换为天，防止网络遇到极大数值导致梯度爆炸 (NaN)
        y2.append(time_diff_seconds / 86400.0)

    # 转换为 Tensor
    X_tensor = torch.tensor(X, dtype=torch.long)
    y1_tensor = torch.tensor(y1, dtype=torch.long)
    y2_tensor = torch.tensor(y2, dtype=torch.float32)

    # 划分数据集 (严格按顺序切割)
    total_len = len(X_tensor)
    split1 = int(0.8 * total_len)
    val_split = int(0.1 * split1)

    X_train_val, X_test = X_tensor[:split1], X_tensor[split1:]
    y1_train_val, y1_test = y1_tensor[:split1], y1_tensor[split1:]
    y2_train_val, y2_test = y2_tensor[:split1], y2_tensor[split1:]

    X_train, X_val = X_train_val[:-val_split], X_train_val[-val_split:]
    y1_train, y1_val = y1_train_val[:-val_split], y1_train_val[-val_split:]
    y2_train, y2_val = y2_train_val[:-val_split], y2_train_val[-val_split:]

    # 构建 Dataloader (分类和回归共享相同的特征 X，但标签 y 不同)
    train_loader_c = DataLoader(TensorDataset(X_train, y1_train), batch_size=batch_size, shuffle=True)
    val_loader_c = DataLoader(TensorDataset(X_val, y1_val), batch_size=batch_size, shuffle=False)
    test_loader_c = DataLoader(TensorDataset(X_test, y1_test), batch_size=batch_size, shuffle=False)

    train_loader_r = DataLoader(TensorDataset(X_train, y2_train), batch_size=batch_size, shuffle=True)
    val_loader_r = DataLoader(TensorDataset(X_val, y2_val), batch_size=batch_size, shuffle=False)
    test_loader_r = DataLoader(TensorDataset(X_test, y2_test), batch_size=batch_size, shuffle=False)

    # ---------- 训练分类模型 ----------
    model_c = ActivityPredictor(vocab_size=num_classes, embedding_dim=32, num_hidden=256, num_out=num_classes, pad_idx=PAD_IDX).to(device)
    optimizer_c = optim.Adam(model_c.parameters(), lr=lr)
    criterion_c = nn.CrossEntropyLoss()
    
    accuracy = train_and_evaluate(model_c, train_loader_c, val_loader_c, test_loader_c, criterion_c, optimizer_c, device, is_regression=False, patience=20)
    precisions.append(accuracy)

    # ---------- 训练回归模型 ----------
    model_r = TimePredictor(vocab_size=num_classes, embedding_dim=32, num_hidden=256, pad_idx=PAD_IDX).to(device)
    optimizer_r = optim.Adam(model_r.parameters(), lr=lr)
    criterion_r = nn.MSELoss() # 回归任务训练用 MSE 损失更平滑
    
    mae = train_and_evaluate(model_r, train_loader_r, val_loader_r, test_loader_r, criterion_r, optimizer_r, device, is_regression=True, patience=20)
    mae_results.append(mae)

    print(f"前缀{i:02d} | 预测活动准确率: {accuracy * 100:.2f}% | 剩余时间误差 (MAE): {mae:.2f} 天")

# ================= 5. 保存结果与可视化 =================
results_df = pd.DataFrame({
    'Prefix_Length': prefix_lengths,
    'Accuracy': precisions,
    'Remaining_Time_MAE_Days': mae_results
})
os.makedirs('result', exist_ok=True)
results_df.to_csv('result/helpdesk_dual_tasks.csv', index=False)

# 绘制双轴图表
fig, ax1 = plt.subplots(figsize=(10, 6))

color = 'tab:blue'
ax1.set_xlabel('Prefix Length')
ax1.set_ylabel('Accuracy', color=color)
ax1.plot(prefix_lengths, precisions, marker='o', color=color, label='Accuracy (Activity)')
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  
color = 'tab:red'
ax2.set_ylabel('MAE Error in Days', color=color)  
ax2.plot(prefix_lengths, mae_results, marker='s', color=color, label='MAE (Remaining Time)')
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()  
plt.title('Performance vs Prefix Length (Dual Tasks)')
plt.grid(True, alpha=0.3)
plt.show()