import cv2
import caffe

from common import utils

# IMG = 'input/test_img.jpg'
IMG = 'input/train_img_centered.jpg'
# TARGET_IMG = 'input/test_target_img.jpg'
TARGET_IMG = 'input/target_img_for_centered.jpg'

MODEL_FILE = 'test.prototxt'
WEIGHTS_FILE = 'output/caffenet_train_iter_2000.caffemodel'
IMG_HEIGHT = 227
IMG_WIDTH = 227


def main():
    net = caffe.Net(MODEL_FILE, WEIGHTS_FILE, caffe.TEST)

    img = cv2.imread(IMG)
    target_img = cv2.imread(TARGET_IMG)

    print(img.shape)
    print(target_img.shape)

    img_resized = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
    target_img_resized = cv2.resize(target_img, (IMG_WIDTH, IMG_HEIGHT))

    img_input = utils.img_to_net_input(img_resized)
    target_input = utils.img_to_net_input(target_img_resized)

    net.blobs['image'].data[...] = img_input
    net.blobs['target'].data[...] = target_input

    out = net.forward()

    print(out)


if __name__ == '__main__':
    main()