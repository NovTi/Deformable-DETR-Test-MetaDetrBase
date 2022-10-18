### Data Preparation

#### Pascal VOC for Few-Shot Object Detection

We transform the original Pascal VOC dataset format into MS-COCO format for parsing. The transformed Pascal VOC dataset is available for download at [GoogleDrive](https://drive.google.com/file/d/1JCxJ2lmNX5E4YsvAZnngVZ5hQeJU67tj/view?usp=sharing).


After downloading MS-COCO-style Pascal VOC, please organize them as following:

```
code_root/
└── data/
    ├── voc_fewshot_split1/     # VOC Few-shot dataset
    ├── voc_fewshot_split2/     # VOC Few-shot dataset
    ├── voc_fewshot_split3/     # VOC Few-shot dataset
    └── voc/                    # MS-COCO-Style Pascal VOC dataset
        ├── images/
        └── annotations/
            ├── xxxxx.json
            ├── yyyyy.json
            └── zzzzz.json
```

Similarly, the few-shot datasets for Pascal VOC are also provided in this repo ([`voc_fewshot_split1`](data/voc_fewshot_split1), [`voc_fewshot_split2`](data/voc_fewshot_split2), and [`voc_fewshot_split3`](data/voc_fewshot_split3)). For each class split, there are 10 data setups with different random seeds. In each K-shot (K=1,2,3,5,10) data setup, we ensure that there are exactly K object instances for each novel class. The numbers of base-class object instances vary.

----------

&nbsp;
## Perform Training

Run the code by this by now:
```bash
GPUS_PER_NODE=1 ./tools/run_dist_launch.sh 1 ./scripts/base_train_voc_base1.sh
```

And remember to change the data root config in main.py line 111.
```bash
parser.add_argument('--data_root', default='../dataset/VOC_detr')
```
Data root is the path of the "voc" in the diagram above which contains images, and annotations.

----------

&nbsp;
## License

The implementation codes of Meta-DETR are released under the MIT license.

Please see the [LICENSE](LICENSE) file for more information.

However, prior works' licenses also apply. It is the users' responsibility to ensure compliance with all license requirements.


------------

&nbsp;
## Citation

If you find Meta-DETR useful or inspiring, please consider citing:

```bibtex
@article{Meta-DETR-2022,
  author={Zhang, Gongjie and Luo, Zhipeng and Cui, Kaiwen and Lu, Shijian and Xing, Eric P.},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence}, 
  title={{Meta-DETR}: Image-Level Few-Shot Detection with Inter-Class Correlation Exploitation}, 
  year={2022},
  doi={10.1109/TPAMI.2022.3195735},
}
```

----------
&nbsp;
## Acknowledgement

Our proposed Meta-DETR is heavily inspired by many outstanding prior works, including [DETR](https://github.com/facebookresearch/detr) and [Deformable DETR](https://github.com/fundamentalvision/Deformable-DETR).
Thank the authors of above projects for open-sourcing their implementation codes!
