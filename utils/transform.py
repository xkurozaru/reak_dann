import random

import torch
from PIL import Image
from torchvision import transforms as T


class ImageTransform:
    def __init__(self, input_size=512, phase="train"):
        if phase == "train":
            self.data_transform = T.Compose(
                [
                    T.RandomResizedCrop(input_size, (0.5, 0.75), (3 / 4, 4 / 3)),
                    RandomRotation90(),
                    T.RandomHorizontalFlip(),
                    T.ToTensor(),
                ]
            )
        elif phase == "test":
            self.data_transform = T.Compose(
                [
                    T.Resize(input_size),
                    T.CenterCrop(input_size),
                    T.ToTensor(),
                ]
            )

    def __call__(self, img):
        return self.data_transform(img)


class RandomRotation90:
    def __call__(self, x):
        i = random.randint(0, 3)
        if isinstance(x, Image.Image):
            x = T.RandomRotation((90 * i, 90 * i))(x)
        elif isinstance(x, torch.Tensor):
            x = torch.rot90(x, i, [1, 2])
        else:
            raise TypeError(f"{type(x)} is unexpected type.")
        return x
