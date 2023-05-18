import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import accuracy_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, RandomSampler
from torchvision.datasets import ImageFolder
from tqdm import tqdm

from models.efficientnet_dann import EfficientNetV2
from utils.seed import set_seed
from utils.transform import ImageTransform

warnings.simplefilter("ignore")
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
BATCH_SIZE = 16
NUM_EPOCH = 50
LR = 1e-3


def main():
    """Leakなし + DA"""
    set_seed()
    dataset = ImageFolder(
        "/data2/eto/Dataset/eggplant_fewclass/7class_leak/train",
        ImageTransform(),
    )
    dataset_tgt = ImageFolder(
        "/data2/eto/Dataset/eggplant_fewclass/7class_leak/test",
        ImageTransform(),
    )
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        num_workers=os.cpu_count(),
        shuffle=True,
        pin_memory=True,
    )
    dataloader_tgt = DataLoader(
        dataset_tgt,
        batch_size=BATCH_SIZE,
        num_workers=os.cpu_count(),
        sampler=RandomSampler(dataset_tgt, num_samples=len(dataset)),
        pin_memory=True,
    )

    device = torch.device("cuda")
    model = EfficientNetV2(len(dataset.classes))
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion_class = nn.CrossEntropyLoss()
    criterion_domain = nn.BCEWithLogitsLoss()

    history = {"acc": [], "class_loss": [], "domain_loss": [], "auc": []}

    loader_len = len(dataloader)
    model.train()
    for epoch in range(NUM_EPOCH):
        print(f"Epoch: {epoch+1} / {NUM_EPOCH}")
        running_labels = {"true": [], "pred": []}
        running_domains = {"true": [], "pred": []}
        running_loss = {"class": 0.0, "domain": 0.0}
        i = 0
        for (inputs_src, labels), (inputs_tgt, _) in tqdm(zip(dataloader, dataloader_tgt), total=len(dataloader_tgt)):
            total_steps = NUM_EPOCH * loader_len
            p = float(i + epoch * loader_len) / total_steps
            alpha = 2.0 / (1.0 + np.exp(-10 * p)) - 1
            i += 1

            optimizer.zero_grad(set_to_none=True)
            inputs_src = inputs_src.to(device, non_blocking=True)
            inputs_tgt = inputs_tgt.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            domains_src = torch.zeros(len(inputs_src)).to(device, non_blocking=True)
            domains_tgt = torch.ones(len(inputs_tgt)).to(device, non_blocking=True)

            outputs_c, outputs_d = model(inputs_src, alpha)
            loss_c = criterion_class(outputs_c, labels)
            loss_d = criterion_domain(outputs_d, domains_src)

            _, output_tgt = model(inputs_tgt, alpha)
            loss_tgt = criterion_domain(output_tgt, domains_tgt)

            loss = loss_c + loss_d + loss_tgt
            loss.backward()
            optimizer.step()

            _, preds = torch.max(outputs_c, 1)
            pred_domains = torch.sigmoid(torch.cat((outputs_d, output_tgt), dim=0))
            domains = torch.cat((domains_src, domains_tgt), dim=0)

            running_labels["pred"].extend(preds.tolist())
            running_labels["true"].extend(labels.tolist())
            running_domains["pred"].extend(pred_domains.tolist())
            running_domains["true"].extend(domains.tolist())
            running_loss["class"] += loss_c.item()
            running_loss["domain"] += loss_d.item() + loss_tgt.item()

        source_acc = accuracy_score(running_labels["true"], running_labels["pred"])
        auc = roc_auc_score(running_domains["true"], running_domains["pred"])
        class_loss = running_loss["class"] / len(dataloader)
        domain_loss = running_loss["domain"] / (len(dataloader) + len(dataloader_tgt))
        print(f"SourceAcc: {source_acc:.4f} AUC: {auc:.4f}")
        print(f"ClassLoss: {class_loss:.4f} DomainLoss: {domain_loss:.4f}")
        history["acc"].append(source_acc)
        history["auc"].append(auc)
        history["class_loss"].append(class_loss)
        history["domain_loss"].append(domain_loss)

    model_path = "./save4/model.pth"
    torch.save(model.state_dict(), model_path)

    fig1, ax1 = plt.subplots()
    ax1.plot(history["acc"], label="source_accuracy")
    ax1.set_xlim(0, NUM_EPOCH)
    ax1.set_ylim(0, 1.0)
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("accuracy")
    ax1.legend()
    fig1.savefig("./save4/accuracy")
    fig2, ax2 = plt.subplots()
    ax2.plot(history["class_loss"], label="class_loss")
    ax2.plot(history["domain_loss"], label="domain_loss")
    ax2.set_xlabel("epoch")
    ax2.set_ylabel("loss")
    ax2.set_xlim(0, NUM_EPOCH)
    ax2.set_ylim(0, 1.0)
    ax2.legend()
    fig2.savefig("./save4/loss")
    fig3, ax3 = plt.subplots()
    ax3.plot(history["auc"], label="domain_AUC")
    ax3.set_xlim(0, NUM_EPOCH)
    ax3.set_ylim(0, 1.0)
    ax3.set_xlabel("epoch")
    ax3.set_ylabel("AUC")
    ax3.legend()
    fig3.savefig("./save4/auc")


if __name__ == "__main__":
    main()
