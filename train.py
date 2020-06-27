import os
import numpy as np
import cv2
import yaml
from mrcnn.config import Config
from mrcnn import utils
import mrcnn.model as modellib
from PIL import Image
from mrcnn import visualize


class FruitAndVegetableConfig(Config):

    # Give the configuration a recognizable name
    NAME = "FruitAndVegetable"

    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 2
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 4  # background + 3 shapes

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 640

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


def file_name(file_dir):
    for root, dirs, files in os.walk(file_dir):
        print(root)  # 当前目录路径
        print(dirs)  # 当前路径下所有子目录
        print(files)  # 当前路径下所有非目录子文件


# 将数据分成训练集和验证集，
def split_data(data_set, seed, m=100, k=10):
    val = []
    train = []
    # random.seed(seed),指定seed的话，每次后面的随机数产生的都是一样的顺序
    np.random.seed(seed)
    for data in data_set:
        if np.random.randint(0,m) <= k:
            val.append(data)
        else:
            train.append(data)
    return val, train


if __name__ == '__main__':
    ROOT_DIR = os.path.abspath("./")
    print(ROOT_DIR)
    MODEL_DIR = 'logs'
    COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
    if not os.path.exists(COCO_MODEL_PATH):
        utils.download_trained_weights(COCO_MODEL_PATH)

    config = FruitAndVegetableConfig()
    config.display()

    dataset_path = os.path.join(ROOT_DIR, "train_data")
    dir_list = os.listdir(dataset_path)
    if ".DS_Store" in dir_list:
        dir_list.remove(".DS_Store")


    dataset = []
    for dir in dir_list:
        dir_path = dataset_path + "/" + dir + "/data"
        dir_sup_list = os.listdir(dir_path)
        dir_sup_list_abs = [os.path.join(dir_path, i) for i in dir_sup_list]
        dataset += dir_sup_list_abs

    val, train = split_data(dataset, 5)

    train_count = len(train)
    val_count = len(val)
    width = 640
    height = 480

    # Training dataset
    dataset_train = FruitAndVegetable()
    dataset_train.load_shapes(train_count, height, width, train)
    dataset_train.prepare()

    # Validation dataset
    dataset_val = FruitAndVegetable()
    dataset_val.load_shapes(val_count, height, width, val)
    dataset_val.prepare()


    # image_ids = np.random.choice(dataset_train.image_ids, 4)
    # for image_id in image_ids:
    #     image = dataset_train.load_image(image_id)
    #     mask, class_ids = dataset_train.load_mask(image_id)
    #     visualize.display_top_masks(image, mask, class_ids, dataset_train.class_names)

    model = modellib.MaskRCNN(mode="training", config=config,
                              model_dir=MODEL_DIR)
    init_with = "coco"  # imagenet, coco, or last

    if init_with == "imagenet":
        model.load_weights(model.get_imagenet_weights(), by_name=True)
    elif init_with == "coco":
        # Load weights trained on MS COCO, but skip layers that
        # are different due to the different number of classes
        # See README for instructions to download the COCO weights
        model.load_weights(COCO_MODEL_PATH, by_name=True,
                           exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",
                                    "mrcnn_bbox", "mrcnn_mask"])
    elif init_with == "last":
        # Load the last model you trained and continue training
        # model.load_weights(model.find_last(), by_name=True)
        model.load_weights("/logs/mask_rcnn_FAV.h5", by_name=True)

    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=4,
                layers='heads')

    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=8,
                layers='4+',
                )

    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE / 10,
                epochs=16,
                layers="all")

    model_path = os.path.join(MODEL_DIR, "mask_rcnn_FAV.h5")
    model.keras_model.save_weights(model_path)