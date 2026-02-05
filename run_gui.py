import argparse
import sys
from pathlib import Path

from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import Qt
from timm.models import create_model
import torch

from gui import gui_main
from UCL.modeling import register_model  # noqa: F401 - needed for model registry side effects


def _default_paths():
    root = Path(__file__).resolve().parent
    return {
        "model": root / "UCL" / "ckpts" / "icoloritv2lab_base_patch16_224_henc6_patchloss.pth",
        "demo_image": root / "UCL" / "assets" / "flower.jpg",
        "save_dir": root / "outputs" / "gui",
        "icon": root / "gui" / "icon.png",
    }


def get_args():
    defaults = _default_paths()
    parser = argparse.ArgumentParser('Region-Specific Colorization UI', add_help=False)
    # Directories
    parser.add_argument('--model_path', type=str, default=str(defaults["model"]), help='checkpoint path of model')
    parser.add_argument('--val_data_path', default=str(defaults["demo_image"]), type=str, help='input image path')
    parser.add_argument('--save_dir', default=str(defaults["save_dir"]), type=str, help='where GUI results are saved')
    parser.add_argument('--device', default='cpu', help='device to use for testing')

    # Dataset parameters
    parser.add_argument('--input_size', default=224, type=int, help='images input size for backbone')

    # Model parameters
    parser.add_argument('--model', default='icoloritv2lab_base_patch16_224_henc6', type=str, help='Name of model to vis')
    parser.add_argument('--drop_path', type=float, default=0.0, help='Drop path rate (default: 0.1)')
    parser.add_argument('--use_rpb', action='store_true', help='relative positional bias')
    parser.add_argument('--no_use_rpb', action='store_false', dest='use_rpb')
    parser.set_defaults(use_rpb=False)
    parser.add_argument('--avg_hint', action='store_true', help='avg hint')
    parser.add_argument('--no_avg_hint', action='store_false', dest='avg_hint')
    parser.set_defaults(avg_hint=True)
    parser.add_argument('--head_mode', type=str, default='cnn', help='head_mode')
    parser.add_argument('--mask_mode', action='store_true', help='mask_mode')
    parser.add_argument('--mode', type=str, default='gt', help='gt or user')

    # Experiments parameters
    parser.add_argument('--local', action='store_true', help='local attention masking')
    parser.add_argument('--grid', action='store_true', help='draw grid map')

    args = parser.parse_args()
    args.save_dir = str(Path(args.save_dir))
    return args


def get_model(args):
    print(f"Creating model: {args.model}")
    model = create_model(
        args.model,
        pretrained=False,
        drop_path_rate=args.drop_path,
        drop_block_rate=None,
        use_rpb=args.use_rpb,
        avg_hint=args.avg_hint,
        head_mode=args.head_mode,
        max_hint_len=None,
    )
    return model


def main():
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning)
    args = get_args()
    model = get_model(args)

    checkpoint = torch.load(args.model_path, map_location=args.device)
    state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.eval()

    app = QApplication(sys.argv)
    ui = gui_main.UclUi(
        color_model=model,
        img_file=args.val_data_path,
        load_size=args.input_size,
        win_size=1080,
        device=args.device,
        args=args,
    )
    icon_path = _default_paths()["icon"]
    if icon_path.exists():
        ui.setWindowIcon(QIcon(str(icon_path)))
    ui.setWindowTitle('Region-Specific Colorization')
    ui.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()

