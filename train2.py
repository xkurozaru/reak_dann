import os
import warnings

import matplotlib.pyplot as plt
import torch
from sklearn.metrics import accuracy_score
from torch import nn
from tqdm import tqdm

from models.efficientnet import EfficientNetV2
from utils.dataset import DANNDataSet
from utils.seed import set_seed
from utils.transform import ImageTransform

warnings.simplefilter("ignore")
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
BATCH_SIZE = 32
NUM_EPOCH = 50
LR = 1e-3


def main():
    """Leakあり"""
    set_seed()
    dataset = DANNDataSet(
        "/data2/eto/Dataset/eggplant_fewclass/7class_leak/train",
        "/data2/eto/Dataset/eggplant_fewclass/7class_leak/leak",
        ImageTransform(),
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        num_workers=os.cpu_count(),
        shuffle=True,
        pin_memory=True,
    )
    device = torch.device("cuda")
    model = EfficientNetV2(len(dataset.classes))
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    history = {"acc": [], "class_loss": []}

    model.train()
    for epoch in range(NUM_EPOCH):
        print(f"Epoch: {epoch+1} / {NUM_EPOCH}")
        running_labels = {"true": [], "pred": []}
        running_loss = 0.0

        for inputs, labels, _ in tqdm(dataloader):
            optimizer.zero_grad(set_to_none=True)
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, preds = torch.max(outputs, 1)
            running_labels["pred"].extend(preds.tolist())
            running_labels["true"].extend(labels.tolist())
            running_loss += loss.item()

        source_acc = accuracy_score(running_labels["true"], running_labels["pred"])
        class_loss = running_loss / len(dataloader)
        print(f"SourceAcc: {source_acc:.4f}")
        print(f"ClassLoss: {class_loss:.4f}")
        history["acc"].append(source_acc)
        history["class_loss"].append(class_loss)

    model_path = "./save2/model.pth"
    torch.save(model.state_dict(), model_path)

    fig1, ax1 = plt.subplots()
    ax1.plot(history["acc"], label="source_accuracy")
    ax1.set_xlim(0, NUM_EPOCH)
    ax1.set_ylim(0, 1.0)
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("accuracy")
    ax1.legend()
    fig1.savefig("./save2/accuracy")
    fig2, ax2 = plt.subplots()
    ax2.plot(history["class_loss"], label="class_loss")
    ax2.set_xlim(0, NUM_EPOCH)
    ax2.set_ylim(0, 1.0)
    ax2.set_xlabel("epoch")
    ax2.set_ylabel("loss")
    ax2.legend()
    fig2.savefig("./save2/loss")


if __name__ == "__main__":
    main()
