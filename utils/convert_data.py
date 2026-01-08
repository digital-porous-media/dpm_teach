import numpy as np
import tifffile
from pathlib import Path
from skimage.io import imread
import matplotlib.pyplot as plt

if __name__ == "__main__":
    datapath = Path(r"C:\Users\bchan\Documents\dpm_teach_data")
    sample_name = "segmented_lrc32_512.ubc"
    # img = imread(datapath / sample_name)
    img = np.fromfile(datapath / "converted_originals" / sample_name, dtype=np.uint8).reshape((512, 512, 512))
    img = (img == 0).astype(np.uint8)
    plt.imshow(img[10])
    plt.colorbar()
    plt.show()
    # plt.savefig(datapath/ "synthetic_vugs.png")
    
    tifffile.imwrite(datapath / "converted" / "sandpack.tif", img)