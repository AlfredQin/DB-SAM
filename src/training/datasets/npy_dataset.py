
import os
import numpy as np
from typing import List, Tuple, Callable
from PIL import Image
from torchvision.transforms import transforms as torch_transforms
from torch.utils.data import Dataset
from super_gradients.training.datasets import DirectoryDataSet
from super_gradients.common.registry.registry import register_dataset
from super_gradients.training.transforms.transforms import SegColorJitter, SegRandomRotate, SegRandomFlip, SegRandomRescale, SegCropImageAndMask

from MedSAM.segment_anything.utils.transforms import ResizeLongestSide


@register_dataset()
class SAMEncEmbeddingDataset(DirectoryDataSet):
    def __init__(
            self,
            root: str,
            embeddings_sub_directory: str,
            targets_sub_directory: str,
            target_extension: str,
            embedding_loader: Callable = None,
            target_loader: Callable = None,
            collate_fn: Callable = None,
            embedding_extensions=('.npy',),
            target_transform: Callable = None,
            add_box_perturbation: bool = False,
    ):
        super().__init__(
            root=root,
            samples_sub_directory=embeddings_sub_directory,
            targets_sub_directory=targets_sub_directory,
            target_extension=target_extension,
            sample_loader=embedding_loader,
            target_loader=target_loader,
            collate_fn=collate_fn,
            sample_extensions=embedding_extensions,
            target_transform=target_transform,
        )
        self.add_box_perturbation = add_box_perturbation

    def __getitem__(self, item):
        embedding, mask = super().__getitem__(item)
        y_indices, x_indices = np.where(mask > 0)
        x_min, x_max = np.min(x_indices), np.max(x_indices)
        y_min, y_max = np.min(y_indices), np.max(y_indices)
        if self.add_box_perturbation:
            # add perturbation to bounding box coordinates
            H, W = mask.shape
            x_min = max(0, x_min - np.random.randint(0, 20))
            x_max = min(W, x_max + np.random.randint(0, 20))
            y_min = max(0, y_min - np.random.randint(0, 20))
            y_max = min(H, y_max + np.random.randint(0, 20))
        box = np.array([x_min, y_min, x_max, y_max])

        return {'embedding': embedding, 'box': box, 'mask_shape': np.array(mask.shape)}, mask[None, ...]


@register_dataset()
class SAMImageNpyDataset(DirectoryDataSet):
    def __init__(
            self,
            root: str,
            samples_sub_directory: str,
            targets_sub_directory: str,
            target_extension: str,
            sample_loader: Callable = None,
            target_loader: Callable = None,
            collate_fn: Callable = None,
            sample_extensions=('.npy',),
            target_transform: Callable = None,
            add_box_perturbation: bool = False,
            perturbation_value=20,
    ):
        super().__init__(
            root=root,
            samples_sub_directory=samples_sub_directory,
            targets_sub_directory=targets_sub_directory,
            target_extension=target_extension,
            sample_loader=sample_loader,
            target_loader=target_loader,
            collate_fn=collate_fn,
            sample_extensions=sample_extensions,
            target_transform=target_transform,
        )

        self.add_box_perturbation = add_box_perturbation
        self.perturbation_value = perturbation_value
        self.transforms = ResizeLongestSide(1024)

    def __getitem__(self, item):
        image, mask = super().__getitem__(item)
        y_indices, x_indices = np.where(mask > 0)
        x_min, x_max = np.min(x_indices), np.max(x_indices)
        y_min, y_max = np.min(y_indices), np.max(y_indices)
        if self.add_box_perturbation:
            # add perturbation to bounding box coordinates
            H, W = mask.shape
            x_min = max(0, x_min - np.random.randint(0, 20))
            x_max = min(W, x_max + np.random.randint(0, 20))
            y_min = max(0, y_min - np.random.randint(0, 20))
            y_max = min(H, y_max + np.random.randint(0, 20))
        box = np.array([x_min, y_min, x_max, y_max])

        # Resize image from 256x256 to 1024x1024, same for box
        image = self.transforms.apply_image(image)
        box = self.transforms.apply_boxes(box, mask.shape)

        return {'images': image, 'boxes': box, 'mask_shape': np.array(mask.shape)}, mask[None, ...]


# test SAMEncEmbedding
if __name__ == '__main__':
    # dataset = SAMEncEmbeddingDataset(
    #     root='/home/qinc/Code/Segmentation/MedSAM/data/Tr_npy',
    #     embeddings_sub_directory='npy_embs',
    #     targets_sub_directory='npy_gts',
    #     target_extension='.npy',
    #     embedding_loader=np.load,
    #     target_loader=np.load,
    #     embeddings_extensions=('.npy',),
    # )
    dataset = SAMImageNpyDataset(
        root='/home/qinc/Dataset/MedSAM/MedSAM-Train',
        sample_loader=lambda x: np.load(x)["img"],
        target_loader=np.load,
        samples_sub_directory='npy_images',
        targets_sub_directory='npy_gts',
        target_extension='.npy',
        sample_extensions='.npz',
    )

    for data in dataset:
        print("ok")
