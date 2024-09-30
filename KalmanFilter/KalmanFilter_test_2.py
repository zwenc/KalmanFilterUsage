import numpy as np
import matplotlib.pyplot as plt
from filterpy.kalman import KalmanFilter

# 初始化参数
real_a = 0.6
real_b = 5.0
point_num = 1000

real_x = np.linspace(0, 100, point_num)
change_a_index = int(point_num / 1.5)
change_a = real_a * 2
yyy = real_x[int( change_a_index)] * real_a + real_b
change_b = yyy - (real_x[int( change_a_index)] * change_a)

# real_y = real_x * real_a + real_b + np.random.normal(0, 0.5, size=point_num)
real_y = []
real_a_tmp = []
real_b_tmp = []
for i in range(point_num):
    if i <= point_num / 5:
        tmp = real_x[i] * real_a + real_b + np.random.normal(0, 0.5)
        real_a_tmp.append(real_a)
        real_b_tmp.append(real_b)
    elif i <= change_a_index:
        tmp = real_x[i] * real_a + real_b + np.random.normal(0, 2)
        real_a_tmp.append(real_a)
        real_b_tmp.append(real_b)
    else:
        tmp = real_x[i] * change_a + change_b + np.random.normal(0, 0.7)
        real_a_tmp.append(change_a)
        real_b_tmp.append(change_b)
    
    real_y.append(tmp)
real_y = np.array(real_y)

# 卡尔曼滤波算法
## 创建卡尔曼滤波器
kf = KalmanFilter(dim_x=2, dim_z=1)

## 初始化状态向量
kf.x = np.array([[0], [0]])     # 初始值
kf.P *= 1000                    # 初始协方差矩阵
kf.F = np.eye(2)                # 状态转移矩阵
kf.H = np.array([[0, 1]])       # 观测矩阵
kf.R = np.array([[0.5 * 20]])    # 观测噪声协方差   (这个值越大，越不信任观测结果，但是当出现一个距离概率分布中心很远的值后，会以观测值为准)
kf.Q = np.eye(2) * 0.0001       # 过程噪声协方差

## 储存估计值
estimates = []
estimates_y = []

## 卡尔曼滤波过程
for index, x in enumerate(real_x):
    y_observed = real_y[index]
    kf.H = np.array([[x, 1]])   # 观测矩阵，y = ax + b
    
    kf.predict()                # 预测步骤
    kf.update(y_observed)       # 更新步骤

    tmp = kf.x.copy()
    estimates.append(tmp)       # 存储估计值
    estimates_y.append(tmp[0] * x + tmp[1])

estimates = np.array(estimates)

# 可视化结果
plt.plot(real_x, real_y, 'o', label='Observed Data', alpha=0.5)
plt.plot(real_x, estimates_y, label='Kalman Filter Estimated', color='blue')
plt.plot(real_x, estimates[:, 0], label='Kalman Filter Estimated a (slope)', color='red')
plt.plot(real_x, estimates[:, 1], label='Kalman Filter Estimated b (slope)', color='black')
plt.plot(real_x, real_b_tmp, color='green', linestyle='--', label='True b')
plt.plot(real_x, real_a_tmp, color='purple', linestyle='--', label='True a')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Kalman Filter')
plt.legend()
plt.show()