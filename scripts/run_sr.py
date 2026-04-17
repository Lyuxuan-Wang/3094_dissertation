import csv
import sys
from pathlib import Path
import cv2

pr = Path(__file__).parent.parent
sys.path.append(str(pr))

from pipeline.degradations.Bicubic import upsampling
from pipeline.models.edsr import run_edsr

SCRIPTS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPTS_DIR.parent
METADATA_DIR = PROJECT_ROOT / "metadata"
RESULTS_CSV = METADATA_DIR / "results.csv"

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
    "experiment_name",
    ]

sys.path.insert(0, str(SCRIPTS_DIR))

def append_csv(row: dict) -> None:
    METADATA_DIR.mkdir(parents=True, exist_ok=True)
    write_header = not RESULTS_CSV.exists() or RESULTS_CSV.stat().st_size == 0
    with open(RESULTS_CSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)

def run_sr(
        dataset_csv: Path,
        output_dir: Path,
        basicsr_root: Path,
        yml_path: Path,
        tile: int = 128,
        tile_pad: int = 10,
        experiment_name: str = "edsr_x4",
) -> None:
    dataset_csv = Path(dataset_csv)
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

        if "GT" not in rows or "LR" not in rows:
            continue

        gt_row = rows["GT"]
        lq_row = rows["LR"]
        gt_path = Path(gt_row["path"])
        lq_path = Path(lq_row["path"])
        stem = Path(lq_row["filename"]).stem

        # Bicubic SR
        bicubic_out_dir = output_dir / "bicubic" / scene
        bicubic_out_dir.mkdir(parents=True, exist_ok=True)
        bicubic_out = bicubic_out_dir / (stem + "_bicubic.png")
        lq = cv2.imread(str(lq_path))
        cv2.imwrite(str(bicubic_out), upsampling(lq))
        append_csv({
            "id": scene + "_" + view_id + "_bicubic",
            "scene": scene,
            "view_id": view_id,
            "input_type": "LR",
            "scale": lq_row.get("scale"),
            "sigma": lq_row.get("sigma"),
            "spp": lq_row.get("spp"),
            "model":"bicubic",
            "task": "SR",
            "filename": bicubic_out.name,
            "path": str(bicubic_out),
            "experiment_name": experiment_name,
        })

        # EDSR SR
        edsr_out_dir = output_dir / "edsr" / scene
        edsr_out_dir.mkdir(parents=True, exist_ok=True)
        edsr_out = edsr_out_dir / (stem + "_edsr.png")
        run_edsr(
            basicsr_root=basicsr_root,
            yml_path=yml_path,
            lq_image=lq_path,
            gt_image=gt_path,
            export_path=edsr_out,
            tile=tile,
            tile_pad=tile_pad,
            exp_name=experiment_name,
        )
        append_csv({
            "id": scene + "_" + view_id + "_edsr",
            "scene": scene,
            "view_id": view_id,
            "input_type": "LR",
            "scale": lq_row.get("scale"),
            "sigma": lq_row.get("sigma"),
            "spp": lq_row.get("spp"),
            "model":"edsr",
            "task": "SR",
            "filename": edsr_out.name,
            "path": str(edsr_out),
            "experiment_name": experiment_name,
        })
        print(f"[OK]" + scene + "view " + view_id)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_csv")
    parser.add_argument("output_dir")
    parser.add_argument("basicsr_root")
    parser.add_argument("yml_path")
    parser.add_argument("--experiment_name", default="edsr_x4")
    parser.add_argument("--tile", type=int, default=128)
    parser.add_argument("--tile_pad", type=int, default=10)
    args = parser.parse_args()
    run_sr(
        args.dataset_csv, args.output_dir,
        args.basicsr_root, args.yml_path,
        args.tile, args.tile_pad, args.experiment_name
    )
