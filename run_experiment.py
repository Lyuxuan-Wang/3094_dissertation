from pipeline.models.edsr import run_edsr
from pipeline.models.ffdnet import FFDNet
from pathlib import Path

# run all scripts
res = run_edsr(
    basicsr_root=Path("models/BasicSR"),
    yml_path=Path("models/BasicSR/options/test/EDSR/test_EDSR_Lx4.yml"),
    export_path=Path("results/edsr"),
    tile=128,
    tile_pad=10,
    exp_name="test_EDSR_Lx4"
)

print(res)

runner = FFDNet(
    model_name="ffdnet_color",
    model_path=Path("models/KAIR/model_zoo/ffdnet_color.pth"),
    input_dir=Path("data/noisy_sigma25"),
    export_dir=Path("results/ffdnet"),
)

runner.run_ffdnet(noise_level=15)