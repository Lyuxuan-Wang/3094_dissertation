import csv
import sys
from pathlib import Path
import numpy as np
from PIL import Image

pr = Path(__file__).parent.parent
sys.path.append(str(pr))

from pipeline.models.ffdnet import FFDNet
from pipeline.CBM3D import denoise as CBM3D

# write metadata while running
# denoise
# MC 16 / 32
# Gaussian： σ = 10 / 25

SCRIPTS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPTS_DIR.parent
METADATA_DIR = PROJECT_ROOT / "metadata"
RESULTS_CSV = METADATA_DIR / "results.csv"

FFDNET_PATH = PROJECT_ROOT / "models/KAIR/model_zoo/ffdnet_color.pth"

CSV_FIELDS = [
    "id",
    "scene",
    "view_id",
    "input_type",
    "scale",
    "sigma",
    "spp",
    "model",
    "task",
    "filename",
    "path",
    "experiment_name"
]
sys.path.insert(0, str(PROJECT_ROOT))

def append_csv(row: dict) -> None:
    METADATA_DIR.mkdir(parents=True, exist_ok=True)
    write_head = not RESULTS_CSV.exists()
    with open(RESULTS_CSV, mode="a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if write_head:
            writer.writeheader()
        writer.writerow(row)

def run_denoise(dataset_csv: Path, output_dir: Path, experiment_name: str = "denoise") -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    groups = {}
    with open(dataset_csv, newline="") as f:
        for row in csv.DictReader(f):
            key = (row["scene"], row["view_id"])
            if key not in groups:
                groups[key] = {}
            groups[key][row["type"]] = row

    for key in groups:
        scene = key[0]
        view_id = key[1]
        rows = groups[key]

        # Gaussian noise
        if "GN" in rows:
            gn_row = rows["GN"]
            noisy = np.array(Image.open(gn_row["path"]))
            sigma = int(gn_row["sigma"])
            stem = Path(gn_row["filename"]).stem

            # CBM3D
            cbm3d_out_dir = output_dir / "cbm3d" / scene / ("gaussian_sigma_" + str(sigma))
            cbm3d_out_dir.mkdir(parents=True, exist_ok=True)
            cbm3d_out_path = cbm3d_out_dir / (stem + "_CBM3D.png")
            denoised = (np.clip(CBM3D(noisy, sigma), 0, 1) * 255).astype(np.uint8)
            Image.fromarray(denoised).save(str(cbm3d_out_path))
            append_csv({
                "id": scene + "_" + view_id + "_cbm3d_gn" + str(sigma),
                "scene": scene,
                "view_id": view_id,
                "input_type": "GN",
                "scale": "-",
                "sigma": sigma,
                "spp": "-",
                "model": "cbm3d",
                "task": "denoise",
                "filename": cbm3d_out_path.name,
                "path": str(cbm3d_out_path),
                "experiment_name": experiment_name
            })

            # FFDNet
            ffdnet_out_dir = output_dir / "ffdnet" / scene / ("gaussian_sigma_" + str(sigma))
            runner = FFDNet(
                model_name="ffdnet_color",
                model_path=FFDNET_PATH,
                input_dir=Path(gn_row["path"]).parent,
                export_dir=ffdnet_out_dir,
            )
            runner.run_ffdnet(noise_level=sigma)
            ffdnet_out_path = ffdnet_out_dir / (stem + "_FFDNet_denoised.png")
            append_csv({
                "id": scene + "_" + view_id + "_ffdnet_gn" + str(sigma),
                "scene": scene,
                "view_id": view_id,
                "input_type": "GN",
                "scale": "-",
                "sigma": sigma,
                "spp": "-",
                "model": "ffdnet",
                "task": "denoise",
                "filename": ffdnet_out_path.name,
                "path": str(ffdnet_out_path),
                "experiment_name": experiment_name
            })

        # Render noise
        for spp in [16, 32]:
            rn_row = None
            for r in rows.values():
                if r["type"] == "RN" and str(r["spp"]) == str(spp):
                    rn_row = r
                    break
            if rn_row is None:
                continue

            noisy = np.array(Image.open(rn_row["path"]))
            stem = Path(rn_row["filename"]).stem

            # CBM3D
            cbm3d_out_dir = output_dir / "cbm3d" / scene / ("render_" + str(spp) + "spp")
            cbm3d_out_dir.mkdir(parents=True, exist_ok=True)
            cbm3d_out_path = cbm3d_out_dir / (stem + "_CBM3D.png")
            denoised = (np.clip(CBM3D(noisy, spp), 0, 1) * 255).astype(np.uint8)
            Image.fromarray(denoised).save(str(cbm3d_out_path))
            append_csv({
                "id": scene + "_" + view_id + "_cbm3d_rn" + str(spp),
                "scene": scene,
                "view_id": view_id,
                "input_type": "RN",
                "scale": "-",
                "sigma": "-",
                "spp": spp,
                "model": "cbm3d",
                "task": "denoise",
                "filename": cbm3d_out_path.name,
                "path": str(cbm3d_out_path),
                "experiment_name": experiment_name
            })

            # FFDNet
            ffdnet_out_dir = output_dir / "ffdnet" / scene / ("render_" + str(spp) + "spp")
            runner = FFDNet(
                model_name="ffdnet_color",
                model_path=FFDNET_PATH,
                input_dir=Path(rn_row["path"]).parent,
                export_dir=ffdnet_out_dir,
            )
            runner.run_ffdnet(noise_level=spp)
            ffdnet_out_path = ffdnet_out_dir / (stem + "_FFDNet_denoised.png")
            append_csv({
                "id": scene + "_" + view_id + "_ffdnet_rn" + str(spp),
                "scene": scene,
                "view_id": view_id,
                "input_type": "RN",
                "scale": "-",
                "sigma": "-",
                "spp": spp,
                "model": "ffdnet",
                "task": "denoise",
                "filename": ffdnet_out_path.name,
                "path": str(ffdnet_out_path),
                "experiment_name": experiment_name
            })
        print("[OK]" + scene + "view" + view_id)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Run FFDNet and CBM3D denoising')
    parser.add_argument("dataset_csv", help="Path to the dataset csv")
    parser.add_argument("output_dir", help="Path to the output directory")
    parser.add_argument("--experiment_name", default="denoise", help="Name of the experiment")
    args = parser.parse_args()

    run_denoise(args.dataset_csv, args.output_dir, args.experiment_name)