import numpy as np
import matplotlib.pyplot as plt

# 生成已知信号和噪声
np.random.seed(0)
n_samples = 500
t = np.arange(n_samples)
# 真实信号 (正弦波)
d = np.sin(0.1 * t) + np.cos(0.3 * t)  
# 添加随机噪声
noise = np.random.normal(0, 0.5, n_samples)
x = d + noise  # 测量信号

# RLS 自适应滤波器参数
order = 32  # 滤波器阶数
delta = 0.1  # 初始化的正则化参数
lambda_ = 0.99  # 衰减因子

# 初始化权重和协方差矩阵
W = np.zeros(order)  # 权重初始化
P = np.eye(order) / delta  # 协方差矩阵初始化
y = np.zeros(n_samples)  # 初始化输出信号

# RLS 自适应滤波过程
for n in range(order, n_samples):
    x_n = x[n-order:n][::-1]  # 当前输入信号（反转顺序）
    y[n] = np.dot(W, x_n)  # 输出信号
    error = d[n] - y[n]  # 计算误差
    
    # 更新协方差矩阵
    pi = P @ x_n  # 计算中间变量
    gain = pi / (lambda_ + x_n @ pi)  # 计算增益
    W += gain * error  # 更新权重
    P = (P - np.outer(gain, pi)) / lambda_  # 更新协方差矩阵

print(W)
# 绘制结果
plt.figure(figsize=(12, 6))
plt.plot(t, d, label='real signal', color='green')
plt.plot(t, x, label='Observed signal (noise)', color='gray', alpha=0.5)
plt.plot(t, y, label='Rectify data', color='blue')
plt.title('RLS Filter 32')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid()
plt.show()

