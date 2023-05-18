import os
import warnings

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
from sklearn import metrics
from torchvision.datasets import ImageFolder
from tqdm import tqdm

from models.efficientnet_dann import DCNN
from utils.seed import set_seed
from utils.transform import ImageTransform

warnings.simplefilter("ignore")
os.environ["CUDA_VISIBLE_DEVICES"] = "6"
BATCH_SIZE = 32


def main():
    set_seed()
    dataset = ImageFolder("/data2/eto/Dataset/eggplant_fewclass/7class_leak/test", ImageTransform(phase="test"))
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        num_workers=os.cpu_count(),
        shuffle=True,
        pin_memory=True,
    )

    classes = ["00_HEAL", "01_PowM", "02_GryM", "06_LefM", "11_LefS", "18_VerW", "19_BacW"]

    device = torch.device("cuda")
    model = DCNN(len(classes))
    model.load_state_dict(torch.load("save5/model.pth"))
    model = model.to(device)

    y_true = []
    y_pred = []

    model.eval()
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader):
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            outputs, _ = model(inputs)
            _, preds = torch.max(outputs, 1)
            y_true.extend(labels.tolist())
            y_pred.extend(preds.tolist())

    report = metrics.classification_report(
        y_true,
        y_pred,
        target_names=classes,
        digits=3,
        output_dict=True,
    )
    plt.figure()
    sns.heatmap(pd.DataFrame(report).iloc[:-1, :].T, annot=True, fmt=".3f")
    plt.tight_layout()
    plt.savefig("./save5/score.png")

    plt.figure(figsize=(15, 10))
    plt.rcParams["font.size"] = 24
    cm = metrics.confusion_matrix(y_true, y_pred, normalize="true")
    cm = pd.DataFrame(data=cm, index=classes, columns=classes)
    sns.heatmap(cm, annot=True, cbar=False, square=True, cmap="BuGn", fmt=".3f")
    plt.tight_layout()
    plt.savefig("./save5/confusion.png")


if __name__ == "__main__":
    main()
