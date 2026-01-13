import matplotlib.pyplot as plt
import numpy as np



def plotInfoTwo(pn, pn2, title, label1, label2, xlabel):
    x = np.array([i for i in range(0,5)])
    # Рисуем график
    plt.plot(x, pn, c='red', label=label1)
    plt.plot(x, pn2, c='blue', label=label2)
    plt.xlabel(xlabel)
    plt.ylabel(title)
    plt.legend()
    plt.title(title)
    plt.show()

def read(path):
    data = []
    with open(path, 'r') as f:
        for line in f:
            try:
                data.append(float(line.strip())) # strip() removes newline/whitespace characters
            except ValueError:
                pass # Skips blank lines or lines that cannot be converted
    return data

if __name__ == '__main__':
    pn = read('/home/rustam/ProjectMy/artifacts/Encoders/pnClass/logs.txt')
    pn2 =read('/home/rustam/ProjectMy/artifacts/Encoders/PointNet2class/res.txt')

    plotInfoTwo(pn, pn2, 'Accuracy of classification', 'pointNet', 'PointNet++', 'epoch')
