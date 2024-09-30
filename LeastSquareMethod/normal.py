import numpy as np
import matplotlib.pyplot as plt

# 初始化参数
real_a = 0.6
real_b = 2.0
point_num = 1000

real_x = np.linspace(0, 100, point_num)
real_y = real_x * real_a + real_b + np.random.normal(0, 0.5, size=point_num)

# 最小二乘法
A = []
for i in range(point_num):
    A.append([real_x[i], 1])
A = np.matrix(A)
B = np.matrix(real_y)
result = (A.T * A).I * A.T * B.T

leastS_a = result.tolist()[0][0]
leastS_b = result.tolist()[1][0]

leastS_y = []
for i in range(point_num):
    leastS_y.append(real_x[i] * leastS_a + leastS_b)

# 可视化结果
plt.plot(real_x, real_y, 'o', label='Observed Data', alpha=0.5)
plt.plot(real_x, leastS_y, label='Least Square Method Estimated', color='red')
# plt.plot(real_x, estimates[:, 1], label='Kalman Filter Estimated a (slope)', color='red')
plt.axhline(real_b, color='green', linestyle='--', label='True b')
plt.axhline(real_a, color='red', linestyle='--', label='True a')
# plt.plot(x_values, a_ls * x_values + b_ls, label='Least Squares Estimated Line', color='orange', linestyle='--')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Least Square Method')
plt.legend()
plt.show()