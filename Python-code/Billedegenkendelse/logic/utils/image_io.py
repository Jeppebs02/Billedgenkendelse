# logic/utils/image_io.py
import io
from PIL import Image
import numpy as np

def bytes_to_pil(image_bytes: bytes) -> Image.Image:

    return Image.open(io.BytesIO(image_bytes)).convert("RGB")

def bytes_to_rgb_np(image_bytes: bytes) -> np.ndarray:

    with Image.open(io.BytesIO(image_bytes)) as im:
        return np.array(im.convert("RGB"))