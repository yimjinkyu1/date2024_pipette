import numpy as np
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
import joblib
from datetime import datetime
import time
import argparse
import numpy as np
from scipy.stats import spearmanr
import matplotlib.pyplot as plt

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch.nn.functional as F

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="MLP", help="select predict model(LinearR, DecisionT, SVM, MLP)")
parser.add_argument("--hidden_layer", type=int, default=200, help="hidden layer size (default=200)")
parser.add_argument("--max_iter", type=int, default=50000, help="mlp max train iteration (default=50000)")
parser.add_argument("--device", type=str, default="cpu", help="train device (default: cpu)")
parser.add_argument("--origin", type=str, default="yes", help="train device (default: yes)")
args = parser.parse_args()



def split_data(input_df):

    x_df = input_df
    x_df = x_df.drop('exp_id', axis=1)
    x_df = x_df.drop('sub_test_id', axis=1)
    x_df = x_df.drop('b2_jobid', axis=1)
    x_df = x_df.drop('opt_ranking_num', axis=1)
    x_df = x_df.drop('fp16', axis=1)
    x_df = x_df.drop('gpu_mem_avg', axis=1)
    x_df = x_df.drop('gpu_mem_max', axis=1)
    x_df = x_df.drop('gpu_type', axis=1)
    x_df = x_df.drop('elapsed_time_per_iteration', axis=1)
    
    #1) calc mini_batch_size
    mini_value= x_df['gbs'] // x_df['dp'] 
    columns = ['mini_bs']
    mbs_df = pd.DataFrame(mini_value, columns=columns)
    #x_df = x_df.drop('gbs', axis=1)
    x_df = pd.concat([x_df, mbs_df], axis=1)
    
    #2) calc #mirro_batch_size 
    gas_value = x_df['mini_bs'] // x_df['mbs']
    columns = ['gas']
    gas_df = pd.DataFrame(gas_value, columns=columns)
    x_df = pd.concat([x_df, gas_df], axis=1)
    
    #3) calc micro_batch_size compute time 
    mbs_fwd_comp_time = x_df['avg_fwd_comp_time'] / x_df['gas']   
    columns = ['mbs_fwd_comp_time']
    mbs_fwd_comp_time_df = pd.DataFrame(mbs_fwd_comp_time, columns=columns)
    

    x_df = x_df[x_df.columns[:0-9]] 
    x_df = pd.concat([x_df, mbs_df], axis=1)

    y_df = mbs_fwd_comp_time_df['mbs_fwd_comp_time']
    
    return x_df, y_df

json_file_path = 'merge_multi_profile_gpt_v100.json'
df = pd.read_json(json_file_path)
x_df, y_df = split_data(input_df=df)   
x_profile = x_df.to_numpy()
y_profile = y_df.to_numpy()

print(x_profile.shape)
print(y_profile.shape)
y_profile = y_profile.reshape(-1, 1)  # y의 형태를 (샘플 수, 1)로 변환 

real_json_file_path =  'merge_profile_real_result_v100.json'
real_df = pd.read_json(real_json_file_path)
condition = real_df['opt_ranking_num'] == -1
real_df = real_df[condition]
x_real_df, y_real_df = split_data(real_df)
x_real = x_real_df.to_numpy()
y_real = y_real_df.to_numpy()
y_real = y_real.reshape(-1, 1)  # y의 형태를 (샘플 수, 1)로 변환 

X_real_train, X_real_test, y_real_train, y_real_test = train_test_split(x_real, y_real, test_size = 0.2, random_state=123)
X_train, X_test, y_train, y_test = train_test_split(x_profile, y_profile, test_size = 0.2, random_state=123)
scaler = StandardScaler()

if args.origin == "no":
    X_train = np.vstack([X_real_train,X_train])
    y_train = np.vstack([y_real_train,y_train])

# 데이터 생성 (scikit-learn 사용)
#X, y = make_regression(n_samples=1000, n_features=10, noise=0.1, random_state=42)
#print(X.shape)
#print(y.shape)
#y = y.reshape(-1, 1)  # y의 형태를 (샘플 수, 1)로 변환
#print(y.shape)

# 데이터 분할
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

if args.device == "gpu":
    device = torch.device("cuda:4")
else:
    device = torch.device("cpu")

# 데이터 정규화
scaler = StandardScaler()



#X_train = scaler.fit_transform(X_real_train)
#X_test = scaler.transform(X_real_test)

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
X_real_train = scaler.transform(X_real_train)
X_real_test = scaler.transform(X_real_test)

# PyTorch Tensor로 변환
X_train_tensor = torch.FloatTensor(X_train)
y_train_tensor = torch.FloatTensor(y_train)
X_test_tensor = torch.FloatTensor(X_test)
y_test_tensor = torch.FloatTensor(y_test)

X_train_tensor = X_train_tensor.to(device)
y_train_tensor = y_train_tensor.to(device)
X_test_tensor = X_test_tensor.to(device)
y_test_tensor = y_test_tensor.to(device)



# PyTorch Tensor로 변환
X_real_train_tensor = torch.FloatTensor(X_real_train)
y_real_train_tensor = torch.FloatTensor(y_real_train)
X_real_test_tensor = torch.FloatTensor(X_real_test)
y_real_test_tensor = torch.FloatTensor(y_real_test)

X_real_train_tensor = X_real_train_tensor.to(device)
y_real_train_tensor = y_real_train_tensor.to(device)
X_real_test_tensor = X_real_test_tensor.to(device)
y_real_test_tensor = y_real_test_tensor.to(device)



# DataLoader 설정
batch_size = 32
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# MLP 모델 정의 (PyTorch 사용)
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 48)
        self.fc3 = nn.Linear(48, 24)
        self.fc4 = nn.Linear(24, 12)
        self.fc5 = nn.Linear(12, output_dim)

   
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        return x
# 학습
epochs = 5000
learning_rate = 0.1
# 모델 초기화, 손실 함수, 최적화 알고리즘 설정
model = MLP(input_dim=11, hidden_dim=64, output_dim=1).to(device)
criterion = nn.MSELoss()

optimizer = optim.Adam(model.parameters(), lr=learning_rate)

step_size = 1000  # 몇 epoch마다 학습률을 줄일 것인지
gamma = 0.1 # 학습률을 몇 배로 줄일 것인지
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

start_time = datetime.now()
for epoch in range(epochs):
    for batch_X, batch_y in train_loader:
        predictions = model(batch_X)
        loss = criterion(predictions, batch_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    # 학습률 감소 적용
    scheduler.step()

    # Epoch마다 손실 출력
    if (epoch+1) % 10 == 0:
        print(f"Epoch {epoch+1}/{epochs}, fwd Loss: {loss.item():.4f}, Learning Rate: {scheduler.get_last_lr()[0]:.5f}")
end_time = datetime.now()
model_train_time = end_time - start_time
model_train_time_sec = model_train_time.total_seconds()
print(f"fwd time model train time(v100) (sec) : {model_train_time_sec}")

# 테스트 데이터에 대한 평가 (선택적)
with torch.no_grad():
    test_predictions = model(X_test_tensor)
    test_loss = criterion(test_predictions, y_test_tensor)
    print(f"Test fwd Loss: {test_loss.item():.4f}")
    
    
with torch.no_grad():
    test_predictions = model(X_real_test_tensor)
    test_loss_real = criterion(test_predictions, y_real_test_tensor)
    print(f"Test fwd Loss real: {test_loss_real.item():.4f}")
    

predict_y = test_predictions.cpu()
test_y = y_real_test_tensor.cpu()

mae = mean_absolute_error(test_y, predict_y)
print(f"mae : {mae}")

mse = mean_squared_error(test_y, predict_y)
print(f"mse : {mse}")

rmse = np.sqrt(mse)
print(f"rmse : {rmse}")

r2 = r2_score(test_y, predict_y)
print(f"r2 : {r2}")

corr, _ = spearmanr(test_y, predict_y)


plt.scatter(test_y, predict_y)
plt.title(f"Spearman Correlation: {corr:.3f}")
plt.xlabel("Predict_Forward_Time(mbs)")
plt.ylabel("Forward_Time(mbs)")
plt.grid(True)

# 그림 파일로 저장
filename = "mbs_fwd_time_spearman_correlation_plot_origin_v100.png"
plt.savefig(filename)

# 그래프 화면에 표시
plt.show()

print(f"Graph saved as {filename}")

torch.save(model, 'model_predict_fwd_time_v100_torch_origin_{}_{}.pkl'.format(args.origin,args.device))

joblib.dump(scaler, 'standard_scaler_model_real_result_v100_torch_origin_{}_{}.pkl'.format(args.origin,args.device))

