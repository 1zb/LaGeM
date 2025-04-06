# LaGeM: A Large Geometry Model for 3D Representation Learning and Diffusion (ICLR 2025)

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

[OpenReview](https://openreview.net/forum?id=72OSO38a2z)

[arxiv](https://arxiv.org/abs/2410.01295)

[Project page](https://1zb.github.io/LaGeM/)

## Training
16 GPUs (4 GPUs with accum_iter 4)
```bash
torchrun --nproc_per_node=4 main_ae.py \\
        --accum_iter=4 \\
        --model AutoEncoder \\
        --output output/ae \\
        --num_workers 32 --point_cloud_size 8192 \\
        --batch_size 16 --epoch 500 \\
        --warmup_epochs 1 --blr 5e-5 --clip_grad 1
```

## Inference

````bash
python eval.py --pc_path your_point_cloud_saved_in_ply.ply
````
## Pretrained Model
https://huggingface.co/Zbalpha/LaGeM

## Bibtex

```bibtex
@inproceedings{
zhang2025lagem,
title={{LaGeM}: A Large Geometry Model for 3D Representation Learning and Diffusion},
author={Biao Zhang and Peter Wonka},
booktitle={The Thirteenth International Conference on Learning Representations},
year={2025},
url={https://openreview.net/forum?id=72OSO38a2z}
}
```
