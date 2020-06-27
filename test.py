import os
import numpy as np
import cv2
import yaml
from mrcnn.config import Config
from mrcnn import utils
import mrcnn.model as modellib
from PIL import Image
from mrcnn import visualize
import matplotlib.pyplot as plt


class FruitAndVegetableConfig(Config):

    # Give the configuration a recognizable name
    NAME = "FruitAndVegetable"

    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 4  # background + 3 shapes

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 1024

    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (16, 32, 64, 128, 256)  # anchor side in pixels

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 100

    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 200

    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 10


class FruitAndVegetable(utils.Dataset):
    # 得到该图中有多少个实例（物体）
    def get_obj_index(self, image):
        n = np.max(image)
        return n

    # 解析labelme中得到的yaml文件，从而得到mask每一层对应的实例标签
    def from_yaml_get_class(self, image_id):
        info = self.image_info[image_id]
        with open(info['yaml_path']) as f:
            temp = yaml.load(f.read())
            labels = temp['label_names']
            del labels[0]
        return labels

    # 重新写draw_mask
    def draw_mask(self, num_obj, mask, image, image_id):
        info = self.image_info[image_id]
        for index in range(num_obj):
            for i in range(info['width']):
                for j in range(info['height']):
                    at_pixel = image.getpixel((i, j))
                    if at_pixel == index + 1:
                        mask[j, i, index] = 1
        return mask

    # 重新写load_shapes，里面包含自己的自己的类别（苹果：apple 玉米：corn 黄瓜：cucumber 橙子：orange）
    # 并在self.image_info信息中添加了path、mask_path 、yaml_path
    def load_shapes(self, count, height, width, data_path_set):
        """Generate the requested number of synthetic images.
        count: number of images to generate.
        height, width: the size of the generated images.
        """
        # Add classes
        self.add_class("shapes", 1, "apple")
        self.add_class("shapes", 2, "corn")
        self.add_class("shapes", 3, "cucumber")
        self.add_class("shapes", 4, "orange")

        for i in range(count):
            img_path = os.path.join(data_path_set[i], "img.png")
            mask_path = os.path.join(data_path_set[i], "label.png")
            yaml_path = os.path.join(data_path_set[i], "info.yaml")
            self.add_image("shapes", image_id=i, path=img_path,
                           width=width, height=height, mask_path=mask_path, yaml_path=yaml_path)

    # 重写load_mask
    def load_mask(self, image_id):
        """Generate instance masks for shapes of the given image ID.
        """
        global iter_num
        info = self.image_info[image_id]
        count = 1  # number of object
        img = Image.open(info['mask_path'])
        num_obj = self.get_obj_index(img)
        mask = np.zeros([info['height'], info['width'], num_obj], dtype=np.uint8)
        mask = self.draw_mask(num_obj, mask, img, image_id)
        occlusion = np.logical_not(mask[:, :, -1]).astype(np.uint8)
        for i in range(count - 2, -1, -1):
            mask[:, :, i] = mask[:, :, i] * occlusion
            occlusion = np.logical_and(occlusion, np.logical_not(mask[:, :, i]))
        labels = []
        labels = self.from_yaml_get_class(image_id)
        labels_form = []
        for i in range(len(labels)):
            if labels[i].find("apple") != -1:
                # print "box"
                labels_form.append("apple")
            elif labels[i].find("corn") != -1:
                # print "column"
                labels_form.append("corn")
            elif labels[i].find("cucumber") != -1:
                # print "package"
                labels_form.append("cucumber")
            elif labels[i].find("orange") != -1:
                # print "fruit"
                labels_form.append("orange")
        class_ids = np.array([self.class_names.index(s) for s in labels_form])
        return mask, class_ids.astype(np.int32)


class InferenceConfig(FruitAndVegetableConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

def get_ax(rows=1, cols=1, size=8):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.

    Change the default size attribute to control the size
    of rendered images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size * cols, size * rows))
    return ax

if __name__ == '__main__':
    class_names = ['BG', 'apple', 'corn', 'cucumber', "orange"]
    ROOT_DIR = os.path.abspath("./")
    print(ROOT_DIR)
    MODEL_DIR = 'logs'
    inference_config = InferenceConfig()
    model = modellib.MaskRCNN(mode="inference",
                              config=inference_config,
                              model_dir=MODEL_DIR)

    model_path = os.path.join(ROOT_DIR, "mask_rcnn_FAV.h5")
    print("Loading weights from ", model_path)
    model.load_weights(model_path, by_name=True)

    width = 640
    height = 480
    image = cv2.imread("test_data/timg4.jpeg")
    # image = cv2.resize(image, (width,height))
    results = model.detect([image], verbose=0)
    r = results[0]
    area_ratio = []
    for i in range(r["masks"].shape[-1]):
        class_area = np.sum(r["masks"][:, :, i])
        class_ratio = class_area/(image.shape[0] * image.shape[1])
        area_ratio.append(class_ratio)
    visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'],
                                area_ratio=area_ratio)
