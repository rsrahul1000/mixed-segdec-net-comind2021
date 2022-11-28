from enum import Enum
import numpy as np
import os
import pickle
from data.dataset import Dataset
from contextlib import redirect_stdout

# _CLASSNAMES = [
#     "bottle",
#     "cable",
#     "capsule",
#     "carpet",
#     "grid",
#     "hazelnut",
#     "leather",
#     "metal_nut",
#     "pill",
#     "screw",
#     "tile",
#     "toothbrush",
#     "transistor",
#     "wood",
#     "zipper",
# ]


def read_split(num_segmented: int, fold: int, kind: str):
    fn = f"DAGM/split_{num_segmented}.pyb"
    with open(f"splits/{fn}", "rb") as f:
        train_samples, test_samples = pickle.load(f)
        if kind == 'TRAIN':
            return train_samples[fold - 1]
        elif kind == 'TEST':
            return test_samples[fold - 1]
        else:
            raise Exception('Unknown')


_CLASSNAMES = [
    "Class1",
    "Class2",
    "Class3",
    "Class4",
    "Class5",
    "Class6",
    "Class7",
    "Class8",
    "Class9",
    "Class10",
    "Class11",
    "Class12",
    "Class13",
    "Class14",
    "Class15",
]


class DatasetSplit(Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"


class MVTecDataset(Dataset):
    def __init__(self, kind: str, cfg):
        super(MVTecDataset, self).__init__(os.path.join(
            cfg.DATASET_PATH, f"Class{cfg.FOLD}"), cfg, kind)
        if kind == "TRAIN":
            self.split = DatasetSplit.TRAIN
        if kind == "TEST":
            self.split = DatasetSplit.TEST
        if kind == "VAL":
            self.split = DatasetSplit.TEST

        # self.split = DatasetSplit.TRAIN

        self.source = cfg.DATASET_PATH
        self.train_val_split = 1.0
        classname = None

        self.classnames_to_use = [f"Class{cfg.FOLD}"] if cfg.FOLD is not None else _CLASSNAMES
        self.read_contents()

    def get_image_data(self):
        imgpaths_per_class = {}
        maskpaths_per_class = {}

        split_train_test = [DatasetSplit.TRAIN, DatasetSplit.TEST]

        for classname in self.classnames_to_use:
            classpath = os.path.join(self.source, classname, self.split.value)
            maskpath = os.path.join(self.source, classname, "ground_truth")
            anomaly_types = os.listdir(classpath)

            imgpaths_per_class[classname] = {}
            maskpaths_per_class[classname] = {}

            for anomaly in anomaly_types:
                anomaly_path = os.path.join(classpath, anomaly)
                anomaly_files = sorted(os.listdir(anomaly_path))
                imgpaths_per_class[classname][anomaly] = [os.path.join(anomaly_path, x) for x in anomaly_files]

                # if self.train_val_split < 1.0:
                #     n_images = len(imgpaths_per_class[classname][anomaly])
                #     train_val_split_idx = int(n_images * self.train_val_split)
                #     if self.split == DatasetSplit.TRAIN:
                #         imgpaths_per_class[classname][anomaly] = imgpaths_per_class[classname][anomaly][:train_val_split_idx]
                #     elif self.split == DatasetSplit.VAL:
                #         imgpaths_per_class[classname][anomaly] = imgpaths_per_class[classname][anomaly][train_val_split_idx:]

                # if self.split == DatasetSplit.TEST and anomaly != "good":
                if anomaly != "good":
                    anomaly_mask_path = os.path.join(maskpath, anomaly)
                    anomaly_mask_files = sorted(os.listdir(anomaly_mask_path))
                    maskpaths_per_class[classname][anomaly] = [os.path.join(anomaly_mask_path, x) for x in anomaly_mask_files]
                else:
                    maskpaths_per_class[classname]["good"] = None

        # Unrolls the data dictionary to an easy-to-iterate list.
        data_to_iterate = []
        for classname in sorted(imgpaths_per_class.keys()):
            for anomaly in sorted(imgpaths_per_class[classname].keys()):
                for i, image_path in enumerate(imgpaths_per_class[classname][anomaly]):
                    data_tuple = [classname, anomaly, image_path]
                    # if self.split == DatasetSplit.TEST and anomaly != "good":
                    if anomaly != "good":
                        data_tuple.append(maskpaths_per_class[classname][anomaly][i])
                    else:
                        data_tuple.append(None)
                    data_to_iterate.append(data_tuple)

        return imgpaths_per_class, data_to_iterate

    def read_contents(self):
        pos_samples, neg_samples = [], []

        self.imgpaths_per_class, self.data_to_iterate = self.get_image_data()
        # samples = read_split(self.cfg.NUM_SEGMENTED, self.cfg.FOLD, self.kind)

        # print(self.imgpaths_per_class, self.data_to_iterate)

        sub_dir = self.kind.lower()

        for class_name, anomaly, image_path, segmented_mask in self.data_to_iterate:
            image = self.read_img_resize(image_path, self.grayscale, self.image_size)
            image_name_short = os.path.basename(image_path).split('/')[-1]

            # if anomaly != "good" and os.path.exists(str(segmented_mask)):
            if os.path.exists(str(segmented_mask)):
                seg_mask, _ = self.read_label_resize(segmented_mask, self.image_size, dilate=self.cfg.DILATE)
                image = self.to_tensor(image)
                seg_loss_mask = self.distance_transform(seg_mask, self.cfg.WEIGHTED_SEG_LOSS_MAX, self.cfg.WEIGHTED_SEG_LOSS_P)
                seg_mask = self.to_tensor(self.downsize(seg_mask, downsize_factor=2))
                seg_loss_mask = self.to_tensor(self.downsize(seg_loss_mask, downsize_factor=2))
                pos_samples.append((image, seg_mask, seg_loss_mask, segmented_mask !=
                                   None, image_path, segmented_mask, image_name_short))

            else:
                # print("is seg not exists: ", image)
                seg_mask = np.zeros_like(image)
                image = self.to_tensor(image)
                seg_loss_mask = self.to_tensor(self.downsize(np.ones_like(seg_mask), downsize_factor=2))
                seg_mask = self.to_tensor(self.downsize(seg_mask, downsize_factor=2))
                neg_samples.append((image, seg_mask, seg_loss_mask, True, image_path, None, image_name_short))

        self.pos_samples = pos_samples
        self.neg_samples = neg_samples
        self.num_pos = len(pos_samples)
        self.num_neg = len(neg_samples)
        print(self.num_pos, self.num_neg)

        with open('mvtec_out.txt', 'a') as f:
            with redirect_stdout(f):
                print("num_pos: ", self.num_pos)
                print("num_neg: ", self.num_neg)
                print("pos_samples: ", self.pos_samples)
                print("neg_samples: ", self.neg_samples)

        self.len = 2 * len(pos_samples) if self.kind in ['TRAIN'] else len(pos_samples) + len(neg_samples)

        self.init_extra()
        # image_path = os.path.join(self.path, sub_dir, image_name)
        # image = self.read_img_resize(
        #     image_path, self.grayscale, self.image_size)
        # img_name_short = image_name[:-4]
        # seg_mask_path = os.path.join(
        #     self.path, sub_dir, "Label",  f"{img_name_short}_label.PNG")

        # if os.path.exists(seg_mask_path):
        #     seg_mask, _ = self.read_label_resize(
        #         seg_mask_path, self.image_size, dilate=self.cfg.DILATE)
        #     image = self.to_tensor(image)
        #     seg_loss_mask = self.distance_transform(
        #         seg_mask, self.cfg.WEIGHTED_SEG_LOSS_MAX, self.cfg.WEIGHTED_SEG_LOSS_P)
        #     seg_mask = self.to_tensor(self.downsize(seg_mask))
        #     seg_loss_mask = self.to_tensor(self.downsize(seg_loss_mask))
        #     pos_samples.append(
        #         (image, seg_mask, seg_loss_mask, is_segmented, image_path, None, img_name_short))

        # else:
        #     seg_mask = np.zeros_like(image)
        #     image = self.to_tensor(image)
        #     seg_loss_mask = self.to_tensor(
        #         self.downsize(np.ones_like(seg_mask)))
        #     seg_mask = self.to_tensor(self.downsize(seg_mask))
        #     neg_samples.append(
        #         (image, seg_mask, seg_loss_mask, True, image_path, seg_mask_path, img_name_short))

        # self.pos_samples = pos_samples
        # self.neg_samples = neg_samples

        # self.num_pos = len(pos_samples)
        # self.num_neg = len(neg_samples)
        # self.len = 2 * \
        #     len(pos_samples) if self.kind in ['TRAIN'] else len(
        #         pos_samples) + len(neg_samples)

        # self.init_extra()
