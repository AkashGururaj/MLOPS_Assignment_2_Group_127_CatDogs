from src.data.preprocess import preprocess_image
from PIL import Image
import os

def test_preprocess(tmp_path):
    # Create a dummy image dynamically
    input_img = tmp_path / "input.jpg"
    img = Image.new("RGB", (300, 300), color="red")
    img.save(input_img)

    output_img = tmp_path / "output.jpg"

    preprocess_image(input_img, output_img)

    assert os.path.exists(output_img)
