import matplotlib.pyplot as plt
import numpy as np



def plotInfoTwo(pn, pn2, title, label1, label2, xlabel):
    x = np.array([i for i in range(0, 410, 10)])
    # Рисуем график
    plt.plot(x, pn, c='red', label=label1)
    plt.plot(x, pn2, c='blue', label=label2)
    plt.xlabel(xlabel)
    plt.ylabel(title)
    plt.legend()
    plt.title(title)
    plt.show()

def plotOne(data, title, label):
    x = np.array([i for i in range(0, 410, 10)])
    # Рисуем график
    plt.plot(x, data, c='red', label=label)
    # plt.plot(x, pn2, c='blue', label=label2)
    # plt.xlabel(xlabel)
    # plt.ylabel(title)
    # plt.legend()
    plt.title(title)
    plt.show()






if __name__ == '__main__':
    pn2_SR = np.loadtxt('/home/rustam/ProjectMy/artifacts/fullModel/PointNetFreezeSeg/res/sr.txt')
    pn_SR = np.loadtxt('/home/rustam/ProjectMy/artifacts/fullModel/pointNetSegmentBucket/res/smallpn_sr.txt')

    pn2_Reward = np.loadtxt('/home/rustam/ProjectMy/artifacts/fullModel/PointNetFreezeSeg/res/reward.txt')
    pn_Reward = np.loadtxt('/home/rustam/ProjectMy/artifacts/fullModel/pointNetSegmentBucket/res/smallpn_reward.txt')

    plotInfoTwo(pn_SR, pn2_SR, 'SuccesRate', 'PointNet', 'PointNetFreeze', 'epoch')
    pn_SR = np.loadtxt('/home/rustam/ProjectMy/artifacts/fullModel/PointNetBaseLine/res/reward.txt')
    # plotOne(pn_SR, 'baseline_reward', 'pointNet')
