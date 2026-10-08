"""Dataset and preprocessing utilities recovered from the TCC notebook."""

import glob
import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class ConvertToHSV_VChannel:
    """Convert an RGB PIL image to the V (brightness) channel of HSV."""

    def __call__(self, pil_image):
        hsv_image = pil_image.convert('HSV')
        _, _, v_channel = hsv_image.split()
        return v_channel


IMAGE_TRANSFORM = transforms.Compose([
    transforms.Resize((512, 512)),
    ConvertToHSV_VChannel(),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])


class AllImagesDataset(Dataset):
    """Labeled image-folder dataset used for ocular disease classification."""

    def __init__(self, root_dir, transform=IMAGE_TRANSFORM):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = []
        self.labels = []
        self.class_names = sorted(
            d for d in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, d))
        )
        self.class_to_idx = {name: i for i, name in enumerate(self.class_names)}

        for class_name, class_idx in self.class_to_idx.items():
            class_dir = os.path.join(root_dir, class_name)
            for ext in ('*.jpg', '*.jpeg', '*.png'):
                files = glob.glob(os.path.join(class_dir, '**', ext), recursive=True)
                self.image_files.extend(files)
                self.labels.extend([class_idx] * len(files))

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image = Image.open(self.image_files[idx]).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, self.labels[idx]


class UnlabeledImagesDataset(Dataset):
    """Unlabeled images used for self-supervised autoencoder training."""

    def __init__(self, root_dir, transform=IMAGE_TRANSFORM):
        self.transform = transform
        self.image_files = []
        for ext in ('*.jpg', '*.jpeg', '*.png'):
            self.image_files.extend(
                glob.glob(os.path.join(root_dir, '**', ext), recursive=True)
            )

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image = Image.open(self.image_files[idx]).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, 0
