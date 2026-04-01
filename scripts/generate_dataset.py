import csv
from pathlib import Path
import cv2
from pipeline.degradations.Bicubic import downsampling
from pipeline.degradations.add_gaussian_noise import add_gaussian_noise

DATA_ROOT = Path("data")
METADATA_PATH = Path("metadata/dataset.csv")

SCALE = 4
GAUSSIAN_SIGMA = 10

def load_img(path):
    return cv2.imread(str(path))

def save_img(img, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img)

def parse_view_id(filename):
    return int(filename.split("_")[1].split(".")[0])

def generate_dataset():
    METADATA_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(METADATA_PATH, "w", newline="") as f:
        writer = csv.writer(f)

        writer.writerow([
            "id",
            "filename",
            "scene",
            "view_id",
            "type",
            "scale",
            "sigma",
            "spp",
            "width",
            "height",
            "path"
        ])

        for scene_dir in DATA_ROOT.iterdir():
            if not scene_dir.is_dir():
                continue

            scene = scene_dir.name
            gt_dir = scene_dir / "GT"

            for gt_path in gt_dir.glob("*.png"):
                filename = gt_path.name
                view_id = parse_view_id(filename)

                # GT
                gt = load_img(gt_path)
                h, w = gt.shape[:2]

                writer.writerow([
                    f"{scene}_{view_id}_GT",
                    filename, scene, view_id,
                    "GT", "-", "-", "-",
                    w, h,
                    str(gt_path)
                ])

                # LR x4
                lr = downsampling(gt, SCALE)
                lr_path = scene_dir / "LR_x4" / filename
                save_img(lr, lr_path)

                writer.writerow([
                    f"{scene}_{view_id}_LR",
                    filename, scene, view_id,
                    "LR", "x4", "-", "-",
                    lr.shape[1], lr.shape[0],
                    str(lr_path)
                ])

                # Gaussian Noise
                noisy = add_gaussian_noise(gt, GAUSSIAN_SIGMA)
                noise_path = scene_dir / "Noise_gaussian_10" / filename
                save_img(noisy, noise_path)

                writer.writerow([
                    f"{scene}_{view_id}_GN",
                    filename, scene, view_id,
                    "GN", "-", GAUSSIAN_SIGMA, "-",
                    w, h,
                    str(noise_path)
                ])

                # Render Noise
                render_noise_path = scene_dir / "Noise_render_32spp" / filename

                if render_noise_path.exists():
                    writer.writerow([
                        f"{scene}_{view_id}_RN",
                        filename, scene, view_id,
                        "RN", "-", "-", "32",
                        w, h,
                        str(render_noise_path)
                    ])