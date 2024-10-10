import numpy as np
import matplotlib.pyplot as plt

# 初始化参数
unit_response = np.array([2.0, 1.0, 0.5, 0.25, 0.125, 0.125/2])      # 系统冲击响应， 假设不发生变化。   思考：如果参数会发生变化怎么办？
point_num = 1000
real_x = np.linspace(0, 100, point_num)         # 采样点位置

y_1 = np.sin(real_x / 2)
y_2 = 2 * np.sin(real_x / 3)

real_input = y_1 + y_2
real_output = []

A_matrix = []
b_matrix = []

input_tmp_buf = np.zeros_like(unit_response)    # 初始化输入buff
for x in real_input:
    input_tmp_buf[0] = x + np.random.normal(0, 0.5)
    A_matrix.append(input_tmp_buf.copy())

    tmp = (unit_response * input_tmp_buf).sum() + np.random.normal(0, 0.5)  # 卷积
    real_output.append(tmp)
    b_matrix.append(tmp.copy())

    input_tmp_buf[1::] = input_tmp_buf[0:-1]

# 最小二乘求系统响应
A = np.matrix(A_matrix)
B = np.matrix(b_matrix).T

x = (A.T * A).I * A.T * B
out_response = np.array(x).reshape(unit_response.shape)
print(out_response)
rectify_output = []
#计算矫正后输出响应
input_tmp_buf2 = np.zeros_like(unit_response)    # 初始化输入buff
for x in real_input:
    input_tmp_buf2[0] = x 

    tmp = (out_response * input_tmp_buf2).sum()  # 卷积
    rectify_output.append(tmp)

    input_tmp_buf2[1::] = input_tmp_buf2[0:-1]

# 可视化结果
plt.plot(real_x, real_input, 'o', label='Observed Data input', alpha=0.5)
plt.plot(real_x, real_output, '-', label='Observed Data, output', alpha=0.5)
plt.plot(real_x, rectify_output, '-', label='Rectify Data, output', alpha=1, color='red')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Least Square Method')
plt.legend()
plt.show()