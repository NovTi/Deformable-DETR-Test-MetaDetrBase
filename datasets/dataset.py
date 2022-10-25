import os
import random
from PIL import Image
import torch
import torch.utils.data
from pycocotools.coco import COCO
from pycocotools import mask as coco_mask
from pathlib import Path
import pdb
import json

from .torchvision_datasets import CocoDetection as TvCocoDetection
from util.misc import get_local_rank, get_local_size
import datasets.transforms as T


class DetectionDataset(TvCocoDetection):
    def __init__(self, args, img_folder, ann_file, transforms, support_transforms, return_masks, activated_class_ids,
                is_val, cache_mode=False, local_rank=0, local_size=1, old_ann_file=None):
        self.img_folder = img_folder
        self.info = json.load(open(ann_file))['img_info']
        self.make_data_lst(activated_class_ids)
        # self.is_finetune = is_finetune
        self.activated_class_ids = activated_class_ids
        self._transforms = transforms
        self.prepare = ConvertCocoPolysToMask(return_masks)
        if is_val:
            from pycocotools.coco import COCO
            self.coco = COCO(old_ann_file)
        # self.shot = args.shot

    def __getitem__(self, idx):
        info = self.data_lst[idx]
        image_id = info['image_id']
        img = Image.open(os.path.join(self.img_folder, info['image_name'])).convert('RGB')
        target = info['annotations']
        target = [anno for anno in target if anno['category_id'] in self.activated_class_ids]
        target = {'image_id': image_id, 'annotations': target}
        img, target = self.prepare(img, target)
        if self._transforms is not None:
            img, target = self._transforms(img, target)
        return img, target

    def __len__(self):
        return len(self.data_lst)

    def make_data_lst(self, activated_class_ids):
        self.data_lst = []
        for i in activated_class_ids:
            self.data_lst += self.info[str(i)]
        # self.data_lst = [*set(self.data_lst)]  # think of another way to remove repetition
        

def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks


class ConvertCocoPolysToMask(object):
    def __init__(self, return_masks=False):
        self.return_masks = return_masks

    def __call__(self, image, target):
        w, h = image.size

        image_id = target["image_id"]
        image_id = torch.tensor([image_id])

        anno = target["annotations"]

        anno = [obj for obj in anno if 'iscrowd' not in obj or obj['iscrowd'] == 0]

        boxes = [obj["bbox"] for obj in anno]
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        classes = [obj["category_id"] for obj in anno]
        classes = torch.tensor(classes, dtype=torch.int64)

        if self.return_masks:
            segmentations = [obj["segmentation"] for obj in anno]
            masks = convert_coco_poly_to_mask(segmentations, h, w)

        keypoints = None
        if anno and "keypoints" in anno[0]:
            keypoints = [obj["keypoints"] for obj in anno]
            keypoints = torch.as_tensor(keypoints, dtype=torch.float32)
            num_keypoints = keypoints.shape[0]
            if num_keypoints:
                keypoints = keypoints.view(num_keypoints, -1, 3)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]
        if self.return_masks:
            masks = masks[keep]
        if keypoints is not None:
            keypoints = keypoints[keep]

        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        if self.return_masks:
            target["masks"] = masks
        target["image_id"] = image_id
        if keypoints is not None:
            target["keypoints"] = keypoints

        # for conversion to coco api
        area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno])
        target["area"] = area[keep]
        target["iscrowd"] = iscrowd[keep]

        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])

        return image, target


def make_transforms(image_set):
    """
    Transforms for query images.
    """
    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]

    if image_set == 'train' or "finetune":
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomColorJitter(p=0.3333),
            T.RandomSelect(
                T.RandomResize(scales, max_size=1152),
                T.Compose([
                    T.RandomResize([400, 500, 600]),
                    T.RandomSizeCrop(384, 600),
                    T.RandomResize(scales, max_size=1152),
                ])
            ),
            normalize,
        ])

    if image_set == 'val' or image_set == 'test':
        return T.Compose([
            T.RandomResize([800], max_size=1152),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')


def make_support_transforms():
    """
    Transforms for support images during the training phase.
    For transforms for support images during inference, please check dataset_support.py
    """
    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    scales = [448, 464, 480, 496, 512, 528, 544, 560, 576, 592, 608, 624, 640, 656, 672]

    return T.Compose([
        T.RandomHorizontalFlip(),
        T.RandomColorJitter(p=0.25),
        T.RandomSelect(
            T.RandomResize(scales, max_size=672),
            T.Compose([
                T.RandomResize([400, 500, 600]),
                T.RandomSizeCrop(384, 600),
                T.RandomResize(scales, max_size=672),
            ])
        ),
        normalize,
    ])


def build(args, image_set, activated_class_ids, is_val, old_ann_file=None):
    root = Path(args.data_root)
    if args.dataset_file in ['voc', 'voc_base1', 'voc_base2', 'voc_base3']:
        assert root.exists(), f'provided Pascal path {root} does not exist'
        PATHS = {
            "train": (root / "images", root / "annotations" / 'voc_info.json'),
            "finetune": (root / "images", root / "annotations" / 'voc_info.json'),
            "val": (root / "images", root / "annotations" / 'voc_val_info.json'),
        }
        img_folder, ann_file = PATHS[image_set]
        
    if args.dataset_file in ['coco', 'coco_base']:
        assert root.exists(), f'provided COCO path {root} does not exist'
        PATHS = {
            "train": (root / "train2017", root / "annotations" / 'instances_train2017.json'),
            "finetune": (root / "images", root / "annotations" / 'pascal_trainval0712.json'),
            "val": (root / "val2017", root / "annotations" / 'instances_val2017.json'),
        }
        img_folder, ann_file = PATHS[image_set]
    
    if old_ann_file:
        old_ann_file = root / 'annotations' / 'pascal_test2007.json'

    return DetectionDataset(args, img_folder, ann_file,
                            transforms=make_transforms(image_set),
                            support_transforms=make_support_transforms(),
                            return_masks=False,
                            activated_class_ids=activated_class_ids,
                            is_val=is_val,
                            cache_mode=args.cache_mode,
                            local_rank=get_local_rank(),
                            local_size=get_local_size(),
                            old_ann_file=old_ann_file)
