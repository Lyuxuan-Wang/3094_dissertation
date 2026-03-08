from pipeline.edsr import run_edsr
from pipeline.ffdnet import FFDNet
from pathlib import Path

"""res = run_edsr(
    basicsr_root=Path("models/BasicSR"),
    yml_path=Path("models/BasicSR/options/test/EDSR/test_EDSR_Lx4.yml"),
    lq_path=Path("models/BasicSR/datasets/test_images/LQ"),
    gt_path=Path("models/BasicSR/datasets/test_images/GT"),
    export_path=Path("results/edsr"),
    tile=128,
    tile_pad=10,
    exp_name="test_EDSR_Lx4"
)

print(res)
"""
runner = FFDNet(
    model_name="ffdnet_color",
    model_path=Path("models/KAIR/model_zoo/ffdnet_color.pth"),
    input_dir=Path("models/KAIR/testsets/test_images"),
    export_dir=Path("results/ffdnet"),
)

runner.run_ffdnet(noise_level=15)