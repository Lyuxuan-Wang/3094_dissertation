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
        lq_path: str | Path,
        gt_path: str | Path,
        export_path: str | Path,
        tile = 128,
        tile_pad = 10,
        exp_name: Optional[str] = None
        ) -> EDSRRunResult:
    """
    Run BasicSR EDSR test while dynamically overriding LQ/GT paths and tile settings.

    :param basicsr_root: Path to the cloned BasicSR repo root.
    :param yml_path: Path to a working EDSR yml file.
    :param lq_path: Path to low quality inputs.
    :param gt_path: Path to ground truth inputs.
    :param export_path: Path to export results.
    :param tile, tile_pad: Tiling parameters to avoid GPU OOM for big images.
    :param exp_name: Optional experiment name override.
    :return: Includes where EDSR results are stored.
    """
    basicsr_root = Path(basicsr_root)
    yml_path = Path(yml_path)
    lq_path = Path(lq_path)
    gt_path = Path(gt_path)
    export_path = Path(export_path)

    # Load the YAML to override paths/tile at runtime
    with open(yml_path, "r") as f:
        yml = yaml.load(f, Loader=yaml.FullLoader)

    ds_key = _pick_first_test_dataset_key(yml)
    ds_cfg = yml["datasets"][ds_key]

    # Override dataset roots
    ds_cfg["lq"] = str(lq_path)
    ds_cfg["gt"] = str(gt_path)

    # Override tile settings
    yml["val"]["tile"] = int(tile)
    yml["val"]["tile_pad"] = int(tile_pad)

    if "path" not in yml:
        yml["path"] = {}
    yml["path"]["basicsr_root"] = basicsr_root

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as f:
        yaml.dump(yml, f)
        temp_yml_path = Path(f.name)

    try:
        cmd = ["python", "models/BasicSR/basicsr/test.py", "-opt", str(temp_yml_path)]
        subprocess.run(cmd, cwd=str(basicsr_root), check=True)

        result_root = basicsr_root / "results" / exp_name / "visualization"
        if not result_root.exists():
            raise RuntimeError(f"Cannot find results in {result_root}")

        dataset_name = ds_cfg.get("name", None)
        candidate = result_root / dataset_name if dataset_name else None
        res_dir = candidate

        export_path.mkdir(parents=True, exist_ok=True)
        copied = _copy_images(res_dir, export_path)
        if copied == 0:
            raise RuntimeError(f"Cannot copy results from {result_root} to {export_path}")

        return EDSRRunResult(
            exp_name=exp_name,
            gt_path=gt_path,
            export_path=export_path,
        )
    finally:
        try:
            os.remove(temp_yml_path)
        except OSError:
            pass

def _pick_first_test_dataset_key(opt: Dict[str, Any]) -> str:
    ds = opt.get("datasets", {})
    return next(iter(ds.keys()))

def _copy_images(result_path: Path, export_path: Path) -> int:
    export_path.mkdir(parents=True, exist_ok=True)
    n = 0
    for p in result_path.rglob("*.png"):
        if p.is_file() and p.suffix == ".png":
            shutil.copy2(p, export_path / p.name)
            n += 1
    return n