# Enabling Region-Specific Control via Lassos in Point-Based Colorization (AAAI 2025)

This is the official PyTorch implementation of the paper: **Enabling Region-Specific Control via Lassos in Point-Based Colorization**.

> **Enabling Region-Specific Control via Lassos in Point-Based Colorization**  
> Sanghyeon Lee, Jooyeol Yun, and Jaegul Choo  
> KAIST  
> In AAAI 2025.

Paper: [arXiv:2412.13469](https://arxiv.org/abs/2412.13469)

> **Abstract:** *Point-based interactive colorization techniques allow users to effortlessly colorize grayscale images using user-provided color hints. However, point-based methods often face challenges when different colors are given to semantically similar areas, leading to color intermingling and unsatisfactory results—an issue we refer to as color collapse. The fundamental cause of color collapse is the inadequacy of points for defining the boundaries for each color. To mitigate color collapse, we introduce a lasso tool that can control the scope of each color hint. Additionally, we design a framework that leverages the user-provided lassos to localize the attention masks. The experimental results show that using a single lasso is as effective as applying 4.18 individual color hints and can achieve the desired outcomes in 30% less time than using points alone.*

## Demo 🎨

Try colorizing images yourself with our GUI!

```bash
python run_gui.py
```
*Note: The GUI saves results to `outputs/gui/` by default.*

## Installation

Our code is implemented in Python, PyTorch, and PyQt5.

```bash
pip install -r requirements.txt
```

## Testing

### Inference

You can generate colorization results using the provided inference script.

```bash
cd UCL
python infer.py \
  --model_path ../outputs/your_model.pth \
  --val_data_path ../data/val/images \
  --val_hint_dir ../data/hint \
  --pred_dir ../predictions
```

## Training

Prepare your dataset (e.g., ImageNet) with the standard structure:
```
train
 └ id1
   └ image1.JPEG
   └ ...
val
 └ id1
   └ ...
```

Run the training script:

```bash
cd UCL
python train.py \
  --data_path ../data/train \
  --val_data_path ../data/val \
  --output_dir ../outputs \
  --exp_name my_experiment
```

## Citation

```bibtex
@inproceedings{lee2025enabling,
  title={Enabling Region-Specific Control via Lassos in Point-Based Colorization},
  author={Lee, Sanghyeon and Yun, Jooyeol and Choo, Jaegul},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={39},
  number={5},
  pages={4544--4552},
  year={2025}
}
```

## Acknowledgments
This codebase is based on [iColoriT](https://github.com/pmh9960/iColoriT).
