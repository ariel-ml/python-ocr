"""_summary_"""
from PIL import Image

IM_FILE = "./assets/mrz-9.jpg"

im = Image.open(IM_FILE)

print(im, im.size)

im.rotate(90, expand=True).show()

# im.save("./assets/mrz-9_rotated.jpg")