import numpy as np
from skimage import io
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import pandas as pd
import os


def load_img(path):
    """
    Load an image
    :param path: Absolute path to image
    :return: image
    """
    img = io.imread(path)
    print(f"{path}: dtype={img.dtype}, max={img.max()}")
    if img.ndim == 3 and img.shape[-1] == 4:
        img = img[..., :3]
    if img.dtype == np.uint16:
        img = img / 65535.0
    else:
        img = img / 255.0
    return img

def compute_psnr(gt, pred):
    """
    Compute peak signal-to-noise ratio
    :param gt: Ground truth image
    :param pred: Model output image
    :return: PSNR in dB (higher is better)
    """
    return float(psnr(gt, pred, data_range=1.0))

def compute_ssim(gt, pred):
    """
    Compute structural similarity
    :param gt: Ground truth image
    :param pred: Model output image
    :return: Mean SSIM in [0, 1] (higher is better)
    """
    return float(ssim(gt, pred, data_range=1.0, channel_axis=-1))

def find_GT(
        result_row: pd.Series,
        dataset: pd.DataFrame,
) -> pd.Series | None:
    """
    Find ground truth image
    :param result_row: One row from results.csv
    :param dataset: Full dataset.csv DataFrame
    :return: The first matching ground truth image row
    """
    task = str(result_row["task"]).lower()
    scene = result_row["scene"]
    view_id = result_row["view_id"]

    candidates = dataset[
        (dataset["scene"] == scene) &
        (dataset["view_id"] == view_id)
    ]

    if candidates.empty:
        return None

    if task == "sr":
        gt = candidates[candidates["type"].str.upper() == "GT"]
    else:
        gt = candidates[candidates["type"].str.upper().isin(["GT", "CLEAN"])]

    return gt.iloc[0] if not gt.empty else None

def make_record(
        row: pd.Series,
        psnr: float | None,
        ssim: float | None,
) -> dict:
    return {
        "id": row["id"],
        "filename": row["filename"],

        "scene": row["scene"],
        "view_id": row["view_id"],

        "input_type": row["input_type"],
        "scale": row["scale"],
        "sigma": row["sigma"],
        "spp": row["spp"],

        "model": row["model"],
        "task": row["task"],

        "psnr": round(psnr, 3) if psnr is not None else None,
        "ssim": round(ssim, 3) if ssim is not None else None,
    }

def evaluate(
        result_row: pd.Series,
        dataset: pd.DataFrame,
        dataroot: str = "",
) -> dict:
    def _abs(p):
        return os.path.join(dataroot, p) if dataroot else p

    gt_row = find_GT(result_row, dataset)
    if gt_row is None:
        print("[evaluate] No ground truth image found")
        return make_record(result_row, psnr=None, ssim=None)

    gt_path = _abs(str(gt_row["path"]))
    result_path = _abs(str(result_row["path"]))

    try:
        img_gt = load_img(gt_path)
        img_pred = load_img(result_path)
    except Exception as e:
        print(f"[evaluate] img load failed {e}")
        return make_record(result_row, psnr=None, ssim=None)

    return make_record(
        result_row,
        psnr=compute_psnr(img_gt, img_pred),
        ssim=compute_ssim(img_gt, img_pred),
    )
