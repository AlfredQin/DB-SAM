import os
import random
import h5py
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from einops import repeat

from super_gradients.common.registry.registry import register_dataset
from training.datasets.label_converter.one_hot import one_hot
from models.segmentation_models.segment_anything.utils.transforms import ResizeLongestSide
from super_gradients.training.transforms.transforms import SegResize

id2label = {
    0: "background",
    1: "spleen",
    2: "right kidney",
    3: "left kidney",
    4: "gallbladder",
    5: "liver",
    6: "stomach",
    7: "aorta",
    8: "pancreas"
}


@register_dataset()
class Synapse_dataset(Dataset):
    def __init__(self, base_dir, list_dir, split, transform=None, target_length: int = 256, num_classes: int = 8):
        self.transform = transform  # using transform in torch!
        self.split = split
        self.sample_list = open(os.path.join(list_dir, self.split + '.txt')).readlines()
        self.data_dir = base_dir
        self.resize_img = ResizeLongestSide(target_length=target_length)
        self.num_classes = num_classes
        self.resize_transform = SegResize(target_length, target_length)

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        if self.split == "train":
            slice_name = self.sample_list[idx].strip('\n')
            data_path = os.path.join(self.data_dir, slice_name + '.npz')
            data = np.load(data_path)
            image = data['image'][..., None].repeat(3, axis=-1) * 255  # 256, 256, 3
            image = image.astype(np.uint8)
            data = {"image": Image.fromarray(image), "mask": Image.fromarray(data["label"])}
            resized_data = self.resize_transform(data)  # 512 to 256
            image, mask = np.array(resized_data["image"]), np.array(resized_data["mask"])

            out = []
            for i in range(1, self.num_classes + 1):
                if np.sum(mask == i) > 100:
                    out.append((image, np.uint8(mask == i)))

            return out
        else:
            vol_name = self.sample_list[idx].strip('\n')
            filepath = self.data_dir + "/{}.npy.h5".format(vol_name)
            data = h5py.File(filepath)
            volume_image, volume_label = data['image'][:], data['label'][:]  # N, H, W; N, H, W

            binary_label = {cls_id: {"image": [], "mask": []} for cls_id in range(1, self.num_classes + 1)}
            for i in range(volume_image.shape[0]):
                image = volume_image[i][..., None].repeat(3, axis=-1) * 255
                image = image.astype(np.uint8)
                sample = {'image': Image.fromarray(image), 'mask': Image.fromarray(volume_label[i])}
                resized_data = self.resize_transform(sample)  # 512 to 256
                image, mask = np.array(resized_data["image"]), np.array(resized_data["mask"])
                for cls_id in range(1, self.num_classes + 1):
                    if np.sum(mask == cls_id) > 100:
                        binary_label[cls_id]["image"].append(image)
                        binary_label[cls_id]["mask"].append(np.uint8(mask == cls_id))

            for cls_id in range(1, self.num_classes + 1):
                if len(binary_label[cls_id]["image"]) > 0:
                    binary_label[cls_id]["image"] = np.stack(binary_label[cls_id]["image"], axis=0)  # (N, H, W, 3)
                    binary_label[cls_id]["mask"] = np.stack(binary_label[cls_id]["mask"], axis=0)  # (N, H, W)

            return binary_label


if __name__ == "__main__":
    train_set = Synapse_dataset(base_dir="/home/qinc/Dataset/SAMed/train_npz",
                                list_dir="/home/qinc/Dataset/SAMed/lists_Synapse", split="train", )
    test_set = Synapse_dataset(base_dir="/home/qinc/Dataset/SAMed/test_vol_h5",
                               list_dir="/home/qinc/Dataset/SAMed/lists_Synapse",
                               split="test_vol", )
    # train_imgs_dst_dir = "/home/qinc/Dataset/SAMed/dataset/train/npy_images"
    # train_masks_dst_dir = "/home/qinc/Dataset/SAMed/dataset/train/npy_gts"
    # for idx, data in enumerate(train_set):
    #     for i in range(len(data)):
    #         image, mask = data[i]
    #         # save image as .npz format
    #         np.savez(os.path.join(train_imgs_dst_dir, f"{idx}_{i}.npz"), img=image)
    #         np.save(os.path.join(train_masks_dst_dir, "{}_{}.npy".format(idx, i)), mask)


    # test_imgs_dst_root_dir = "/home/qinc/Dataset/SAMed/dataset/test"
    # for case_id, data in enumerate(test_set):
    #     for cls_id in range(1, 9):
    #         cls_name = id2label[cls_id]
    #         imgs, gts = data[cls_id]["image"], data[cls_id]["mask"]
    #         if len(imgs) > 0:
    #             save_dir = os.path.join(test_imgs_dst_root_dir, f"{cls_id}_{cls_name}")
    #             if not os.path.exists(save_dir):
    #                 os.makedirs(save_dir)
    #             # save as npz format
    #             np.savez(os.path.join(save_dir, f"{case_id}.npz"), imgs=imgs, gts=gts)
