# voc_base1 full shot
import json
from torchvision_datasets import CocoDetection as TvCocoDetection
import pdb


# def parse_args() -> argparse.Namespace:
#     parser = argparse.ArgumentParser(description='Building Annotation')
#     parser.add_argument('--annfile', type=str, required=True, help='annotation file')
#     # parser.add_argument('--debug', type=bool, default=False, help='debug mode')
#     # parser.add_argument('--opts', default=None, nargs=argparse.REMAINDER)
#     args = parser.parse_args()
#     return args

def main():
    test = TvCocoDetection('../../dataset/VOC_detr/images', '../../dataset/VOC_detr/annotations/pascal_trainval0712.json')
    a, b = test[0]
    # c, d = test[1]
    data_path = '../../dataset/VOC_detr/annotations/pascal_trainval0712.json'
    data = json.load(open(data_path))
    pdb.set_trace()


if __name__ == '__main__':
    # args = parse_args()
    main()
