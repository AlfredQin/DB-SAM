__package__ = "MedSAM.dataset"
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from super_gradients.common.registry.registry import register_dataset

from MedSAM.segment_anything.utils.transforms import ResizeLongestSide


class NpzDataset(Dataset):
    def __init__(
            self,
            root,
            add_box_perturbation: bool = False,
            perturbation_value=20,
):
        self.root = root
        self.npz_files = sorted(os.listdir(root))
        self.add_box_perturbation = add_box_perturbation
        self.perturbation_value = perturbation_value
        self.transforms = ResizeLongestSide(1024)
        self.dataset_name = root.split('/')[-1]

    def __len__(self):
        return len(self.npz_files)

    def __getitem__(self, idx):
        npz_file = np.load(os.path.join(self.root, self.npz_files[idx]))
        gts, images = npz_file['gts'], npz_file['imgs']   # gts: (N, H, W), images: (N, H, W, 3)

        boxes = []
        image_list = []
        for img_id in range(len(images)):
            gt = gts[img_id]
            y_indices, x_indices = np.where(gt > 0)   # This is where bug occurs, it should be y_indices, x_indices = np.where(gt > 0)
            x_min, x_max = np.min(x_indices), np.max(x_indices)
            y_min, y_max = np.min(y_indices), np.max(y_indices)
            if self.add_box_perturbation:
                # add perturbation to bounding box coordinates
                H, W = gt.shape
                x_min = max(0, x_min - np.random.randint(9, 10))
                x_max = min(W, x_max + np.random.randint(9, 10))
                y_min = max(0, y_min - np.random.randint(9, 10))
                y_max = min(H, y_max + np.random.randint(9, 10))

            box = np.array([x_min, y_min, x_max, y_max])
            image = self.transforms.apply_image(images[img_id])
            box = self.transforms.apply_boxes(box, gt.shape)
            image_list.append(image)
            boxes.append(box)
        boxes = np.array(boxes)
        images = np.array(image_list)

        return {'images': images, 'boxes': boxes, "original_size": gts.shape[-2:]}, gts, {"dataset_name": self.dataset_name}

    @staticmethod
    def collate_fn(data):
        images = []
        boxes = []
        original_size = []
        gts = []
        for d in data:
            images.append(torch.as_tensor(d[0]['images']))
            boxes.append(torch.as_tensor(d[0]['boxes']))
            original_size.append(torch.as_tensor(d[0]['original_size']))
            gts.append(torch.as_tensor(d[1]))
        return {'images': images, 'boxes': boxes, "original_size": original_size}, gts, data[0][2]


def get_medsam_test_dataset(test_root_dir, add_box_perturbation=False, perturbation_value=20):
    task_names = os.listdir(test_root_dir)
    test_datasets = {}
    for task_name in task_names:
        test_datasets[task_name] = NpzDataset(os.path.join(test_root_dir, task_name),
                                              add_box_perturbation=add_box_perturbation,
                                              perturbation_value=perturbation_value)

    return test_datasets


if __name__ == '__main__':
    import cv2
    # dataset = NpzDataset('/home/qinc/Downloads/Tr_Release_Part1', load_to_memory=True)
    # for data in dataset:
    #     image, gt, bbox = data  # image: (H, W, C), gt: (1, H, W), bbox: (4, )
    #     # Using cv2 to visualize the image and gt
    #     image = image.numpy().astype(np.uint8)
    #     gt = gt.numpy().astype(np.uint8)
    #     # image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    #     image = cv2.rectangle(image, (int(bbox[1]), int(bbox[0])), (int(bbox[3]), int(bbox[2])), (0, 255, 0), 2)
    #     cv2.imshow('image', image)
    #     cv2.imshow('gt', gt[0] * 255)
    #     cv2.waitKey(0)
    #
    #     print()

    # root_dir = '/home/qinc/Downloads/Test/22-2D-Images'
    # file_list = os.listdir(root_dir)
    # for file_name in file_list:
    #     file = np.load(os.path.join(root_dir, file_name))
    #     imgs, gts = file['imgs'], file['gts']
    #     img, gt = file['img'], file['gt']
    #     x_indices, y_indices = np.where(gt > 0)
    #     x_min, x_max = np.min(x_indices), np.max(x_indices)
    #     y_min, y_max = np.min(y_indices), np.max(y_indices)
    #     box = np.array([x_min, y_min, x_max, y_max])
    #     image = cv2.rectangle(img, (int(box[1]), int(box[0])), (int(box[3]), int(box[2])), (0, 255, 0), 2)
    #     image = cv2.resize(image, (1024, 1024))
    #     gt = cv2.resize(gt, (1024, 1024))
    #     cv2.imshow('image', image)
    #     cv2.imshow('gt', gt * 255)
    #     cv2.waitKey(0)
    #     print()

    dataset = NpzDataset('/home/qinc/Downloads/Test/01-MR-T1_Brain_Ventricle')
    for data in dataset:
        print()
