import os

from PIL import Image
from torch.utils import data


def pil_loader(img_path):
    with open(img_path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")


def make_dataset(directory, class_to_idx, domain):
    instances = []
    directory = os.path.expanduser(directory)
    for target_class in sorted(class_to_idx.keys()):
        class_index = class_to_idx[target_class]
        target_dir = os.path.join(directory, target_class)
        if not os.path.isdir(target_dir):
            continue
        for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
            for fname in sorted(fnames):
                path = os.path.join(root, fname)
                item = path, class_index, domain
                instances.append(item)
    return instances


class DANNDataSet(data.Dataset):
    def __init__(self, source, target, transform):
        self.transform = transform
        source_dir = source
        target_dir = target

        classes_s = [d.name for d in os.scandir(source_dir) if d.is_dir()]
        classes_t = [d.name for d in os.scandir(target_dir) if d.is_dir()]
        classes = classes_s + classes_t
        classes = list(set(classes))
        classes.sort()
        self.classes = classes
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        self.class_to_idx = class_to_idx

        self.source_samples = make_dataset(source_dir, class_to_idx, "source")
        self.target_samples = make_dataset(target_dir, class_to_idx, "target")
        self.samples = self.source_samples + self.target_samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        img_path, label, domain = self.samples[index]
        if domain == "source":
            domain = 0.0
        elif domain == "target":
            domain = 1.0
        img = pil_loader(img_path)
        img = self.transform(img)

        return img, label, domain


class LeakDataSet(data.Dataset):
    def __init__(self, source, target, leak, transform):
        self.transform = transform
        source_dir = source
        target_dir = target
        leak_dir = leak

        classes_s = [d.name for d in os.scandir(source_dir) if d.is_dir()]
        classes_t = [d.name for d in os.scandir(target_dir) if d.is_dir()]
        classes = classes_s + classes_t
        classes = list(set(classes))
        classes.sort()
        self.classes = classes
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        self.class_to_idx = class_to_idx

        self.source_samples = make_dataset(source_dir, class_to_idx, "source")
        self.target_samples = make_dataset(target_dir, class_to_idx, "target")
        self.leak_samples = make_dataset(leak_dir, class_to_idx, "leak")
        self.samples = self.source_samples + self.target_samples + self.leak_samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        img_path, label, domain = self.samples[index]
        if domain == "source":
            domain = 0.0
            train = 1.0
        elif domain == "target":
            domain = 1.0
            train = 0.0
        elif domain == "leak":
            domain = 1.0
            train = 1.0
        img = pil_loader(img_path)
        img = self.transform(img)

        return img, label, domain, train
