import matplotlib.pyplot as plt
import numpy as np

# figsize=(7, 5)
fig1 = plt.figure(1)
channel = [100, 125, 150, 175, 200, 225, 250, 275, 300]

# one-hot
f1_l1 = [0.903, 0.939, 0.945, 0.950, 0.951, 0.951, 0.953, 0.955, 0.955]

f1_l2 = [0.573, 0.731, 0.889, 0.929, 0.927, 0.943, 0.945, 0.944, 0.947]

f1_l3 = [0.624, 0.715, 0.825, 0.838, 0.915, 0.921, 0.923, 0.941, 0.917]

# embedding
em_f1_l1 = [0.885, 0.890, 0.896, 0.901, 0.905, 0.906, 0.909, 0.910, 0.911]

em_f1_l2 = [0.888, 0.900, 0.905, 0.910, 0.909, 0.913, 0.913, 0.918, 0.920]

em_f1_l3 = [0.858, 0.882, 0.892, 0.895, 0.899, 0.899, 0.906, 0.900, 0.913]

ax1 = fig1.add_subplot(1, 1, 1, alpha=0.3)  # 可选facecolor='#4f4f4f',alpha=0.3


# plt.yticks(np.linspace(0.76, 0.9, 8))
#
# plt.ylim(0.76, 0.9)

plt.plot(channel, em_f1_l1, 's-', color='#7fc8ff', label="L1")  # color='#7fc8ff'

plt.plot(channel, em_f1_l2, 'o-', label="L2")

plt.plot(channel, em_f1_l3, '^-', color='#99ff00', label="L3")


# plt.plot(channel, acc300, '^-', label="T=300")

# s;.;o;
# plt.scatter(channel, acc300, marker='^')


ax1.legend(loc='lower right')
# 设置标题、x轴和y轴标题、图例文字

# linewidth="3" color="r"
plt.grid(True, linestyle="-")

title = plt.title(u'POS tagging', fontsize=10, color='black')
xlabel = plt.xlabel(u'Channel', fontsize=9, color='black')
ylabel = plt.ylabel(u'F1', fontsize=9, color='black')

# 设置坐标轴的的颜色和文字大小
plt.tick_params(colors='black', labelsize=10)

fig1.savefig('./result_f1.png', bbox_inches='tight', facecolor=fig1.get_facecolor())
# plt.show()
