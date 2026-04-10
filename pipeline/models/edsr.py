from typing import Optional, Dict, Any
import os
import tempfile
import shutil
import subprocess
from pathlib import Path
import yaml
from dataclasses import dataclass

@dataclass
class EDSRRunResult:
    exp_name: str
    gt_path: str | Path
    export_path: str | Path

def run_edsr(
        basicsr_root: str | Path,
        yml_path: str | Path,
        lq_image: str | Path,
        gt_image: str | Path,
        export_path: str | Path,
        tile = 128,
        tile_pad = 10,
        exp_name: Optional[str] = None
        ) -> EDSRRunResult:
    """
    Run BasicSR EDSR test on a single image pair.

    :param basicsr_root: Path to the cloned BasicSR repo root.
    :param yml_path: Path to a working EDSR yml file.
    :param lq_image: Path to a single low quality input image.
    :param gt_image: Path to the corresponding ground truth input image.
    :param export_path: Path to export result image.
    :param tile, tile_pad: Tiling parameters to avoid GPU OOM for big image.
    :param exp_name: Optional experiment name override.
    :return: Includes where EDSR result is stored.
    """
    basicsr_root = Path(basicsr_root)
    yml_path = Path(yml_path)
    lq_image = Path(lq_image)
    gt_image = Path(gt_image)
    export_path = Path(export_path)

    # Load the YAML to override paths/tile at runtime
    with open(yml_path, "r") as f:
        yml = yaml.load(f, Loader=yaml.FullLoader)

    ds_key = _pick_first_test_dataset_key(yml)
    ds_cfg = yml["datasets"][ds_key]

    # Override tile settings
    yml["val"]["tile"] = int(tile)
    yml["val"]["tile_pad"] = int(tile_pad)

    if "path" not in yml:
        yml["path"] = {}
    yml["path"]["basicsr_root"] = str(basicsr_root)

    with tempfile.TemporaryDirectory() as tmp_root:
        tmp_lq = Path(tmp_root) / "lq"
        tmp_gt = Path(tmp_root) / "gt"
        tmp_lq.mkdir()
        tmp_gt.mkdir()

        shutil.copy2(lq_image, tmp_lq / lq_image.name)
        shutil.copy2(gt_image, tmp_gt / gt_image.name)

        ds_cfg["lq"] = str(tmp_lq)
        ds_cfg["gt"] = str(tmp_gt)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as f:
            yaml.dump(yml, f)
            temp_yml_path = Path(f.name)
        try:
            cmd = ["python", "basicsr/test.py", "-opt", str(temp_yml_path)]
            subprocess.run(cmd, cwd=str(basicsr_root), check=True)

            result_root = basicsr_root / "results" / exp_name / "visualization"

            dataset_name = ds_cfg.get("name", None)
            res_dir = result_root / dataset_name if dataset_name else result_root

            result_image = res_dir / lq_image.name
            if not  result_image.exists():
                raise RuntimeError(f"Cannot find result image at {result_image}")

            export_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(result_image, export_path)

        finally:
            try:
                os.remove(temp_yml_path)
            except OSError:
                pass

    return EDSRRunResult(
                    exp_name=exp_name,
                    gt_path=gt_image,
                    export_path=export_path,
                )
def _pick_first_test_dataset_key(opt: Dict[str, Any]) -> str:
    ds = opt.get("datasets", {})
    return next(iter(ds.keys()))
