import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from torchvision.datasets import ImageFolder
from tqdm import tqdm

from models.efficientnet import EfficientNetV2
from utils.seed import set_seed
from utils.transform import ImageTransform

os.environ["CUDA_VISIBLE_DEVICES"] = "3"
warnings.simplefilter("ignore")
BATCH_SIZE = 64


def main():
    set_seed()
    source_dataset = ImageFolder("/data2/eto/Dataset/eggplant_fewclass/7class_leak/train", ImageTransform(phase="test"))
    target_dataset = ImageFolder("/data2/eto/Dataset/eggplant_fewclass/7class_leak/test", ImageTransform(phase="test"))
    leak_dataset = ImageFolder("/data2/eto/Dataset/eggplant_fewclass/7class_leak/leak", ImageTransform(phase="test"))

    source_dataloader = torch.utils.data.DataLoader(
        source_dataset,
        batch_size=BATCH_SIZE,
        num_workers=os.cpu_count(),
        sampler=torch.utils.data.sampler.RandomSampler(source_dataset, num_samples=len(target_dataset)),
        pin_memory=True,
    )
    target_dataloader = torch.utils.data.DataLoader(
        target_dataset,
        batch_size=BATCH_SIZE,
        num_workers=os.cpu_count(),
        shuffle=True,
        pin_memory=True,
    )
    leak_dataloader = torch.utils.data.DataLoader(
        leak_dataset,
        batch_size=BATCH_SIZE,
        num_workers=os.cpu_count(),
        shuffle=True,
        pin_memory=True,
    )
    classes = source_dataset.classes
    print(classes)

    device = torch.device("cuda")
    model = EfficientNetV2(num_class=len(classes))
    model.load_state_dict(torch.load("./save3/model.pth"))
    model = model.to(device)

    X = np.empty((0, 1280), float)
    y = np.empty(0, float)
    z = np.empty(0, float)

    model.eval()
    with torch.no_grad():
        len_dataloader = min(len(source_dataloader), len(target_dataloader))
        source_iter = iter(source_dataloader)
        target_iter = iter(target_dataloader)
        i = 0
        pbar = tqdm(total=len_dataloader)
        while i < len_dataloader:
            source_data = next(source_iter)
            target_data = next(target_iter)
            s_img, s_label = source_data
            t_img, t_label = target_data
            s_domain = torch.zeros(len(s_label))
            t_domain = torch.ones(len(t_label))

            s_img = s_img.to(device, non_blocking=True)
            s_label = s_label.to(device, non_blocking=True)
            t_img = t_img.to(device, non_blocking=True)
            t_label = t_label.to(device, non_blocking=True)
            s_domain = s_domain.to(device, non_blocking=True)
            t_domain = t_domain.to(device, non_blocking=True)

            inputs = torch.cat([s_img, t_img], dim=0)
            labels = torch.cat([s_label, t_label], dim=0)
            domains = torch.cat([s_domain, t_domain], dim=0)

            output = model.feature_extractor(inputs)
            X = np.append(X, output.to("cpu").detach().numpy().copy(), axis=0)
            y = np.append(y, labels.to("cpu").detach().numpy().copy(), axis=0)
            z = np.append(z, domains.to("cpu").detach().numpy().copy(), axis=0)

            i += 1
            pbar.update(1)

        for inputs, labels in leak_dataloader:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            domains = torch.full_like(labels, fill_value=2).to(device, non_blocking=True)

            output = model.feature_extractor(inputs)
            X = np.append(X, output.to("cpu").detach().numpy().copy(), axis=0)
            y = np.append(y, labels.to("cpu").detach().numpy().copy(), axis=0)
            z = np.append(z, domains.to("cpu").detach().numpy().copy(), axis=0)

    pbar.close()
    tsne = TSNE(n_components=2)
    X_reduced = tsne.fit_transform(X)
    print("tsne complete.")
    scaler = StandardScaler()
    X_reduced = scaler.fit_transform(X_reduced)
    print("standard scale complete.")

    plt.figure(figsize=(14, 10))
    plt.scatter(X_reduced[:, 0], X_reduced[:, 1], s=10, c=y, cmap="jet", alpha=0.5)
    plt.rcParams["font.size"] = 24
    plt.axis("off")
    plt.colorbar()
    plt.savefig("./save3/class.png")

    plt.figure(figsize=(14, 10))
    plt.scatter(X_reduced[:, 0], X_reduced[:, 1], s=10, c=z, cmap="jet", alpha=0.5)
    plt.rcParams["font.size"] = 24
    plt.axis("off")
    plt.colorbar()
    plt.savefig("./save3/domain.png")


if __name__ == "__main__":
    main()
