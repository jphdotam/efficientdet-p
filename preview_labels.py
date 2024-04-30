import os

from pycocotools.coco import COCO
import matplotlib.pyplot as plt
import skimage.io as io

# Path to your COCO formatted JSON file
annotation_file = r'E:\Dropbox\Work\Papers\ALGE\data\data_processed\annotations\LGE\instances_train2017.json'
image_root = r"E:\Dropbox\Work\Papers\ALGE\data\data_processed\LGE\train"
coco = COCO(annotation_file)

# Select one image to display
image_ids = coco.getImgIds()
image_data = coco.loadImgs(image_ids[10])[0]  # Load the first image info

print(image_data)

# Load the image
image = io.imread(os.path.join(image_root, image_data['file_name']))  # Adjust this based on how your images are accessed

# Load annotations
annIds = coco.getAnnIds(imgIds=image_data['id'], iscrowd=None)
anns = coco.loadAnns(annIds)
print(anns)

# Plot
plt.imshow(image, cmap='gray')
coco.showAnns(anns, draw_bbox=True)
plt.show()