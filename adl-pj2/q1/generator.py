from torchvision.utils import save_image
from torchvision.datasets import MNIST
from torchvision import datasets, transforms
from torch.utils import data as tdataset
import torch.nn as nn
import torch.nn.functional as F
import torch
import argparse
import os


class LinearAccuracy(nn.Module):
    def __init__(self, encoder):
        super(LinearAccuracy).__init__()
        self.encoder = encoder
        self.linear = nn.Linear()

    def forward(self, imgs):
        gen_latents = self.encoder(imgs)
        gen_latents = self.linear(gen_latents)
        latent_softmax = F.relu(gen_latents)
        return latent_softmax

def img_generator(model, dataloader, real_dir, gen_dir):
    os.makedirs(real_dir, exist_ok=True)
    os.makedirs(gen_dir, exist_ok=True)
    model.eval()
    counter = 0
    for i, (real_imgs, labels) in enumerate(dataloader):
        labels = labels.to(model.device)
        gen_imgs, noise = model.generate_imgs(cls=labels)
        for j in range(gen_imgs.size(0)):
            real_path = os.path.join(real_dir, f'real_{counter}.png')
            gen_path = os.path.join(gen_dir, f'gen_{counter}.png')
            save_image(real_imgs[j], real_path)
            save_image(gen_imgs[j], gen_path)

def get_test_loader(data_path, image_size, batch_size):
    dataset = MNIST(
        data_path,
        download=True,
        train=False,
        transform=transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
        ])
    )
    loader = tdataset.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )
    return loader

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='image and latent generator')
    parser.add_argument('--model_path', type=str, help='path to saved model')
    args = parser.parse_args()
    model = torch.load(args.model_path)
    model.eval()