import numpy as np
import tifffile as tiff
import json
from pathlib import Path
import requests
import matplotlib.pyplot as plt

class DPMDataloader:
    def __init__(self, registry_path: str = "../utils/registry.json", cache_dir: str = "./dpm_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

        with open(registry_path, 'r') as f:
            self.registry = json.load(f)

    def __getitem__(self, sample):
        return self.get_sample(sample)

    def get_sample(self, sample):
        sample_info = self.registry.get(sample)
        if not sample_info:
            raise ValueError(f"Sample '{sample}' not found in registry.")
        sample_path = self.cache_dir / f"{sample}.tif"

        if not sample_path.exists():
            # Download the sample file from the provided URL
            print(f"Downloading sample '{sample}' from {sample_info['url']}")
            sample_url = sample_info["url"]
            response = requests.get(sample_url, stream=True)
            with open(sample_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=1024):
                    if chunk:
                        f.write(chunk)
        else:
            print(f"Loading cached sample '{sample}' from {sample_path}")
        img = tiff.imread(sample_path)
        return img

def load_sample(sample_name: str, **kwargs):
    dataloader = DPMDataloader(**kwargs)
    return dataloader.get_sample(sample_name)


# if __name__ == "__main__":
#     dataloader = DPMDataloader()
#     sample_name = "sandpack"
#     sample = dataloader.get_sample("sample_name")
#     print(f"Shape of {sample_name} image: {sample.shape}")
#     plt.imshow(sample)
#     plt.show()
