import matplotlib.pyplot as plt
import numpy as np



def plotInfo(pn, pn2, title, label1, label2, xlabel):
    x = np.array([i for i in range(0, 410, 10)])
    # Рисуем график
    plt.plot(x, pn, c='red', label=label1)
    plt.plot(x, pn2, c='blue', label=label2)
    plt.xlabel(xlabel)
    plt.ylabel(title)
    plt.legend()
    plt.title(title)
    plt.show()


if __name__ == '__main__':
    pn2_SR = np.loadtxt('/home/rustam/ProjectMy/artifacts/fullModel/PointNet2SegmentBucket/res/sr.txt')
    pn_SR = np.loadtxt('/home/rustam/ProjectMy/artifacts/fullModel/pointNetSegmentBucket/res/smallpn_sr.txt')

    pn2_Reward = np.loadtxt('/home/rustam/ProjectMy/artifacts/fullModel/PointNet2SegmentBucket/res/reward.txt')
    pn_Reward = np.loadtxt('/home/rustam/ProjectMy/artifacts/fullModel/pointNetSegmentBucket/res/smallpn_reward.txt')

    # plotInfo(pn_SR, pn2_SR, 'Succes rate', 'PointNet', 'PointNet++', 'epoch')
    plotInfo(pn_Reward, pn2_Reward, 'Reward', 'PointNet', 'PointNet++', 'epoch')
