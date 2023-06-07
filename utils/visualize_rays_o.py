import matplotlib.pyplot as plt
import torch
import numpy as np
from torch.utils import data

from dataset.load_blender import BlenderDataset_v2

class InfiniteSamplerWrapper(data.sampler.Sampler):

    def __init__(self, num_samples):
        self.num_samples = num_samples

    def __iter__(self):
        return iter(InfiniteSampler(self.num_samples))

    def __len__(self):
        return 2**31


def InfiniteSampler(n):
    order = np.random.permutation(n)
    i = 0
    while True:
        yield order[i]
        i += 1
        if i == n:
            order = np.random.permutation(n)
            i = 0


def get_dataloader(dataset_type, datadir, pseudo_ratio=0.5):
    trainset = BlenderDataset_v2(
        datadir,
        dim_dir=3,
        dim_rgb=1,
        hold_ratio=0,
        pseudo_ratio=-1)
    trainloader = torch.utils.data.DataLoader(
        dataset=trainset,
        batch_size=1024,
        num_workers=1,
        pin_memory=True,
        sampler=InfiniteSamplerWrapper(len(trainset)))
    return iter(trainloader), len(trainset)

datadir_kd = "data/nerf_synthetic/lego_pseudo_images_depth2k"
trainloader, n_total_img = get_dataloader("blender", datadir_kd)
distances_list = []
for i in range(1):
    rays = trainloader.next()
    origins = rays[0]
    reshaped_tensor = origins.view(-1, 3)

    # Split the tensor along the second dimension
    x, y, z = torch.split(reshaped_tensor, 1, dim=1)
    x = x.squeeze().tolist()
    y = y.squeeze().tolist()
    z = z.squeeze().tolist()

    # Compute distances from origin
    distances = []
    for i in range(len(x)):
        distances.append(np.sqrt(x[i] ** 2 + y[i] ** 2 + z[i] ** 2))
    distances_list.extend(distances)

# ax = plt.figure().add_subplot(projection='3d')
# ax.plot(x_list, y_list, z_list, 'o')
plt.plot(list(range(len(distances_list))), distances_list, 'o')
plt.show()
