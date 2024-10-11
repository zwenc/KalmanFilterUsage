import numpy as np
import matplotlib.pyplot as plt

# 生成已知信号和噪声
np.random.seed(0)
n_samples = 500
t = np.arange(n_samples)
# 真实信号 (正弦波)
d = np.sin(0.1 * t)  
# 添加随机噪声
noise = np.random.normal(0, 0.5, n_samples)
x = d + noise  # 测量信号

# LMS 自适应滤波器参数
mu = 0.01  # 学习率
order = 32  # 滤波器阶数
W = np.zeros(order)  # 初始化权重
y = np.zeros(n_samples)  # 初始化输出信号

# 自适应滤波过程
for n in range(order, n_samples):
    x_n = x[n-order:n]  # 当前输入信号
    y[n] = np.dot(W, x_n)  # 输出信号
    error = d[n] - y[n]  # 计算误差
    W += mu * error * x_n  # 更新滤波器权重

# 绘制结果
plt.figure(figsize=(12, 6))
plt.plot(t, d, label='real signal', color='green')
plt.plot(t, x, label='Observed signal (noise)', color='gray', alpha=0.5)
plt.plot(t, y, label='Rectify data', color='blue')
plt.title('LMS Filter')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid()
plt.show()
