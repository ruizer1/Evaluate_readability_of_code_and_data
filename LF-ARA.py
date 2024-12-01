import time

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

def read_and_preprocess(file_path):
    # 读取CSV文件
    data = pd.read_csv(file_path)

    # 选择特征和标签
    X = data.iloc[:, 1:].values  # 假设第一列是类别标签，从第二列开始是特征
    y = data.iloc[:, 0].values

    # print(X)
    # 特征缩放
    scaler = StandardScaler()
    scaler.fit(X)
    X = scaler.transform(X)
    # print(X)

    return X, y
file_path = ''
X, y = read_and_preprocess(file_path)

class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc5 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc5(out)

        return out

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42, shuffle=True)


input_size = X_train.shape[1]
hidden_size = XX  # 可以根据需要调整
num_classes = 5

model = NeuralNetwork(input_size, hidden_size, num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.009)

# 将文本标签转换为数字标签
label_mapping = {'等级1': 0, '等级2': 1, '等级3': 2, '等级4': 3, '等级5': 4}
y_train = np.array([label_mapping[label] for label in y_train])
y_val = np.array([label_mapping[label] for label in y_val])

# 初始化最高准确率变量
best_train_accuracy = 0.0
# 初始化保存模型的变量
best_model = None

def train(model, X_train, y_train, X_val, y_val, epochs):
    global best_train_accuracy, best_model
    best_train_accuracy = 0  # 初始化最佳训练准确率
    best_model = None  # 初始化最佳模型

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        # print(torch.tensor(X_train, dtype=torch.float32))
        output = model(torch.tensor(X_train, dtype=torch.float32))
        loss = criterion(output, torch.tensor(y_train, dtype=torch.long))
        loss.backward()
        optimizer.step()

        # 计算训练集和验证集的准确率

        model.eval()
        with torch.no_grad():
            _, predicted_train = torch.max(output.data, 1)
            train_accuracy = accuracy_score(y_train, predicted_train.numpy())

            val_output = model(torch.tensor(X_val, dtype=torch.float32))
            val_loss = criterion(val_output, torch.tensor(y_val, dtype=torch.long))
            _, predicted_val = torch.max(val_output.data, 1)
            val_accuracy = accuracy_score(y_val, predicted_val.numpy())
            val_recall = recall_score(y_val, predicted_val.numpy(), average='macro')
            val_f1 = f1_score(y_val, predicted_val.numpy(), average='macro')
        # time.sleep(420)
        # 检查是否更新最高准确率和保存模型
        if val_accuracy > best_train_accuracy:
            best_train_accuracy = val_accuracy
            best_model = model.state_dict()

            # 保存模型，文件名包含当前epoch
            # torch.save(best_model, f'best_model_epoch_{epoch+1}_{best_train_accuracy:.4f}.pth')

        print(
            f'Epoch {epoch + 1}/{epochs}, Train Loss: {loss.item()}, '
            f'Train Accuracy: {train_accuracy:.4f}, Val Loss: {val_loss.item()}, '
            f'Val Accuracy: {val_accuracy:.4f}, Val Recall: {val_recall:.4f}, '
            f'Val F1: {val_f1:.4f}')

    # # 如果需要在训练结束后再次保存最后一轮的模型，可以取消注释以下行
    # torch.save(model.state_dict(), f'final_model_epoch_{epoch+1}.pth')

def train_csv(model, X_train, y_train, X_val, y_val, epochs):
    global best_train_accuracy, best_model
    best_val_accuracy = 0.0  # 用于跟踪最高验证集准确率
    best_val_epoch = -1       # 用于记录达到最高准确率的epoch

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        output = model(torch.tensor(X_train, dtype=torch.float32))
        loss = criterion(output, torch.tensor(y_train, dtype=torch.long))
        loss.backward()
        optimizer.step()

        # 评估模型并计算验证集的准确率
        model.eval()
        with torch.no_grad():
            val_output = model(torch.tensor(X_val, dtype=torch.float32))
            _, predicted = torch.max(val_output, 1)
            val_accuracy = accuracy_score(y_val, predicted.numpy())

        # 检查是否达到新的最高验证集准确率
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_val_epoch = epoch

            # 保存验证集的输入、标签和预测结果
            val_inputs = X_val
            val_true_labels = y_val
            val_predicted_labels = predicted.numpy()

            Tra_inputs = X_train
            Tra_true_labels = y_train
            _, Tra_predicted = torch.max(output, 1)
            Tra_predicted_labels = Tra_predicted.numpy()

        # 打印信息
        print(f'Epoch {epoch + 1}/{epochs}, ..., Val Accuracy: {val_accuracy:.4f}')

    # 保存达到最高验证集准确率时的结果到CSV文件
    if best_val_epoch != -1:
        df_val_results = pd.DataFrame({
            'R2': val_inputs[:,0],
            'True_Label': val_true_labels,
            'Predicted_Label': val_predicted_labels
        })
        file_name = f'val_results_best_accuracy_epoch_{best_val_epoch}_2024_11_18.csv'
        df_val_results.to_csv(file_name, index=False)



train_csv(model, X_train, y_train, X_val, y_val, epochs=200)