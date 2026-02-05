# src

Core implementation for training/inference.

Train:
```
python train.py --data_path ../data/train --val_data_path ../data/val --output_dir ../outputs --exp_name my_experiment
```

Inference:
```
python infer.py --model_path ../outputs/your_model.pth --val_data_path ../data/val/images --val_hint_dir ../data/hint --pred_dir ../predictions
```
