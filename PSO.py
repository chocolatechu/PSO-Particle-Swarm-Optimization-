# 先观察一下我们的目标函数的走势
import math
import numpy as np
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import val_function
x = [x/10 for x in range(-50, 50, 1)]
y = [y/10 for y in range(-50, 50, 1)]
x, y = np.meshgrid(x, y)
R1 = (x**2 + y**2)
R2 = -10*np.cos(2*np.pi*x)
R3 = -10*np.cos(2*np.pi*y)
Z = R1+R2+R3+20




# 初始化参数
c1 = 1.49445
c2 = 1.49445
# 进化次数
maxgen = 1000
# 种群规模
sizepop = 100

# 边界
Vmax = 1
Vmin = -1
popmax = 5
popmin = -5

pop = np.zeros([sizepop,2])
V = np.zeros([sizepop,2])
fit = []
# 产生初代粒子与速度
for i in range(sizepop):
    pop[i, 0] = random.random()*5
    pop[i, 1] = random.random()*5
    V[i, 0] = random.random()
    V[i, 1] = random.random()
    # 计算每个粒子的适应度
    fit.append(val_function.val_function(pop[i,:]))

# 计算极值
bestfitness = np.max(fit)
bestindex = fit.index(bestfitness)
# 全局最佳
Gbest = pop[bestindex,:]
# 个体最佳
pbest = pop
# 个体最佳适应度值
fitnessPbest = fit
# 全局最佳适应度值
fitnessGbest = bestfitness
# 开始寻找最优
yy = []
for i in range(maxgen):
    for j in range(sizepop):
        # 速度更新
        V[j, :] = V[j, :] + c1*random.random()*(pbest[j, :] - pop[j, :]) + c2*random.random()*(Gbest - pop[j, :])
        for k in range(len(V[j,:])):
            if V[j, k] > Vmax:
                V[j, k] = Vmax
            elif V[j, k] < Vmin:
                V[j, k] = Vmin

        # 种群更新
        pop[j, :] = pop[j, :] + V[j, :]
        for k in range(len(V[j, :])):
            if pop[j, k] > popmax:
                pop[j, k] = popmax
            elif pop[j, k] < popmin:
                pop[j, k] = popmin

        fit[j] = val_function.val_function(pop[j, :])
    # 更新极值们
    for j in range(sizepop):
        if fit[j] > fitnessPbest[j]:
            pbest[j, :] = pop[j, :]
            fitnessPbest[j] = fit[j]

        if fit[j] > fitnessGbest:
            Gbest = pop[j,:]
            fitnessGbest = fit[j]
    yy.append(fitnessGbest)

print(fitnessGbest, Gbest[0], Gbest[1])
fig = plt.figure()
ax = Axes3D(fig)
ax.plot_surface(x, y, Z, rstride=1, cstride=1, cmap='hot')
ax.scatter(Gbest[0], Gbest[1], fitnessGbest, c='b')

fig1 = plt.figure()
plt.plot(yy)
plt.show()