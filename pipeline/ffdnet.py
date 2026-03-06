import os
from pathlib import Path
from utils import util
import torch
from models.KAIR.models.network_ffdnet import FFDNet as net

class FFDNet:
    def __init__(self, model_name, model_path: str | Path, input_dir: str | Path, export_dir: str | Path):
        self.model_path = model_path
        self.input_dir = input_dir
        self.export_dir = export_dir
        self.model_name = model_name
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if 'color' in model_name:
            self.n_channels = 3  # setting for color image
            self.nc = 96  # setting for color image
            self.nb = 12

        self.model = net(
            in_nc=self.n_channels,
            out_nc=self.n_channels,
            nc=self.nc,
            nb=self.nb,
            act_mode='R'
        )
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        for k, v in self.model.named_parameters():
            v.requires_grad = False
        self.model.to(self.device)

        self.export_dir.mkdir(parents=True, exist_ok=True)

    def denoise_image(self, input_path: str | Path, output_path: str | Path, noise_level: int):
        img_L = util.imread_uint(input_path, n_channels=self.n_channels)
        img_L = util.uint2single(img_L)

        img_L = util.single2tensor4(img_L)
        img_L = img_L.to(self.device)

        sigma = torch.full((1,1,1,1), noise_level/255., device=self.device, dtype=img_L.dtype)

        with torch.no_grad():
            output = self.model(img_L, sigma)

        output = util.tensor2uint(output)
        util.imsave(output, output_path)

    def run_ffdnet(self, noise_level: int):
        img_paths = util.get_image_paths(self.input_dir)
        for img_path in img_paths:
            img_name = os.path.basename(img_path)
            save_path = self.export_dir / img_name

            self.denoise_image(img_path, save_path, noise_level)