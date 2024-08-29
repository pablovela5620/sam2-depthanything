# Sam2 and DepthAnything 2
An easy way to track subjects using Segment Anything 2, and then project them to 3D using depths from DepthAnything 2.
Uses Gradio for an interactive UI, Rerun to visualize, and Pixi for a simple installation

<p align="center">
  <img src="media/sam2+depthanything.gif" alt="example output" width="480" />
</p>

## Install and Run
Make sure you have the [Pixi](https://pixi.sh/latest/#installation) package manager installed
```bash
git clone https://github.com/pablovela5620/sam2-depthanything.git
cd sam2-depthanything
pixi run app
```

All commands can be listed using `pixi task list`
## Hosted Demo
Demos can be found on huggingface spaces (currently broken, works fine locally, will update once fully working!)

<a href='https://huggingface.co/spaces/pablovela5620/sam2-depthanything'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue'></a>

## Acknowledgements
Thanks to the following great works!

[DepthAnything](https://github.com/LiheYoung/Depth-Anything)
```bibtex
@inproceedings{depthanything,
      title={Depth Anything: Unleashing the Power of Large-Scale Unlabeled Data}, 
      author={Yang, Lihe and Kang, Bingyi and Huang, Zilong and Xu, Xiaogang and Feng, Jiashi and Zhao, Hengshuang},
      booktitle={CVPR},
      year={2024}
}
```
[Segment Anything 2](https://github.com/facebookresearch/segment-anything-2)
```bibtex
@article{ravi2024sam2,
  title={SAM 2: Segment Anything in Images and Videos},
  author={Ravi, Nikhila and Gabeur, Valentin and Hu, Yuan-Ting and Hu, Ronghang and Ryali, Chaitanya and Ma, Tengyu and Khedr, Haitham and R{\"a}dle, Roman and Rolland, Chloe and Gustafson, Laura and Mintun, Eric and Pan, Junting and Alwala, Kalyan Vasudev and Carion, Nicolas and Wu, Chao-Yuan and Girshick, Ross and Doll{\'a}r, Piotr and Feichtenhofer, Christoph},
  journal={arXiv preprint arXiv:2408.00714},
  url={https://arxiv.org/abs/2408.00714},
  year={2024}
}
```
