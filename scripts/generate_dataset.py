import csv
from pathlib import Path
import cv2
from pipeline.degradations.Bicubic import downsampling
from pipeline.degradations.add_gaussian_noise import add_gaussian_noise

DATA_ROOT = Path("data")
METADATA_PATH = Path("metadata/dataset.csv")

SCALE = 4
GAUSSIAN_SIGMA = 10
RENDER_SPPS = [16, 32]

def load_img(path):
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"---{path} is not a valid path---")
    return img

def save_img(img, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(img, 'save'):
        img.save(str(path))
    else:
        cv2.imwrite(str(path), img)

def parse_view_id(filename):
    return int(filename.split("_")[1].split(".")[0])

def generate_dataset():
    METADATA_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(METADATA_PATH, "w", newline="") as f:
        writer = csv.writer(f)

        writer.writerow([
            "id",
            "scene",
            "view_id",

            "type",             # GT / LR / Gaussian_noise / render_noise

            "scale",            # x4 (only for LR)
            "sigma",            # Gaussian
            "spp",              # Render noise

            "width",
            "height",

            "filename",
            "path"
        ])

        for scene_dir in DATA_ROOT.iterdir():
            if not scene_dir.is_dir():
                continue

            scene = scene_dir.name
            gt_dir = scene_dir / "GT"

            for gt_path in sorted(gt_dir.glob("*.png")):
                filename = gt_path.name
                view_id = parse_view_id(filename)

                # GT
                gt = load_img(gt_path)
                h, w = gt.shape[:2]

                writer.writerow([
                    f"{scene}_{view_id}_GT", scene, view_id,
                    "GT", "-", "-", "-",
                    w, h, filename, str(gt_path)
                ])

                # LR x4
                lr = downsampling(gt, SCALE)
                lr_path = scene_dir / "LR_x4" / filename
                save_img(lr, lr_path)

                writer.writerow([
                    f"{scene}_{view_id}_LR", scene, view_id,
                    "LR", "x4", "-", "-",
                    lr.shape[1], lr.shape[0], filename, str(lr_path)
                ])

                # Gaussian Noise
                noise_path = scene_dir / f"Noise_gaussian_{GAUSSIAN_SIGMA}" / filename
                save_img(add_gaussian_noise(gt_path, GAUSSIAN_SIGMA), noise_path)

                writer.writerow([
                    f"{scene}_{view_id}_GN", scene, view_id,
                    "GN", "-", GAUSSIAN_SIGMA, "-",
                    w, h, filename,
                    str(noise_path)
                ])

                # Render Noise
                for spp in RENDER_SPPS:
                    render_noise_path = scene_dir / f"Noise_render_{spp}spp" / filename

                    if render_noise_path.exists():
                        writer.writerow([
                            f"{scene}_{view_id}_RN{spp}", scene, view_id,
                            "RN", "-", "-", str(spp),
                            w, h, filename,
                            str(render_noise_path)
                        ])