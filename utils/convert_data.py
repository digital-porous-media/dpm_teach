import numpy as np
import tifffile
from pathlib import Path
from skimage.io import imread
import matplotlib.pyplot as plt
import porespy as ps
from skimage.transform import resize

if __name__ == "__main__":
    # datapath = Path(r"C:\Users\bchan\Documents\dpm_teach_data")
    # sample_name = "segmented_lrc32_512.ubc"
    # # img = imread(datapath / sample_name)
    # img = np.fromfile(datapath / "converted_originals" / sample_name, dtype=np.uint8).reshape((512, 512, 512))
    # img = (img == 0).astype(np.uint8)
    # plt.imshow(img[10])
    # plt.colorbar()
    # plt.show()
    # # plt.savefig(datapath/ "synthetic_vugs.png")
    img = tifffile.imread(Path(r"/mnt/c/Users/bchan/Documents/dpm_teach_data/converted") / "beadpack_slice.tif")
    img = np.where(img < 128, 0, 1).astype(np.uint8)
    tifffile.imwrite(Path(r"/mnt/c/Users/bchan/Documents/dpm_teach_data/converted") / "beadpack_slice.tif", img)
    # print(np.unique(img))
    # plt.imshow(img, cmap="binary")
    # plt.colorbar()
    # plt.show()
    
    # tifffile.imwrite(datapath / "converted" / "sandpack.tif", img)
    # n = 3
    # sierpinski_carpet = ps.generators.sierpinski_foam(shape=[3**n, 3**n], n=n)
    
    # sierpinski_carpet = resize(~sierpinski_carpet, (512, 512), order=0, preserve_range=True).astype(np.uint8)*255

    # # print(np.unique(sierpinski_carpet))
    # tifffile.imwrite(Path(r"/mnt/c/Users/bchan/Documents/dpm_teach_data/converted") / "sierpinski_carpet.tif", sierpinski_carpet)
    # plt.imshow(sierpinski_carpet, cmap="binary")
    # plt.show()