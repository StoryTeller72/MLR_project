import json
import matplotlib.pyplot as plt

# пути к файлам
file1 = "/home/rustam/ProjectMy/artifacts/Encoders/PointNetSeg/logs.json"
file2 = "/home/rustam/ProjectMy/artifacts/Encoders/Pn2seg/new/logs.json"

def load_iou(path):
    with open(path, "r") as f:
        data = json.load(f)

    epochs = [x["epoch"] for x in data]
    miou = [x["mAcc"] for x in data]

    return epochs, miou

# загрузка данных
epochs1, miou1 = load_iou(file1)
epochs2, miou2 = load_iou(file2)

# построение графика
plt.figure()
plt.plot(epochs1, miou1, label="PointNet++")
plt.plot(epochs2, miou2, label="PointNet")

plt.xlabel("Epoch")
plt.ylabel("mAcc")
plt.title("mAcc vs Epoch")
plt.legend()
plt.grid(True)

plt.show()
