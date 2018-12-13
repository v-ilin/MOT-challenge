import numpy as np
import cv2
import os
import pandas as pd

os.environ['GLOG_minloglevel'] = '2'
import caffe

IMG_1 = '1.jpg'
IMG_2 = '2.jpg'
MODEL_FILE = 'tracker.prototxt'
SOLVER_FILE = 'solver.prototxt'
TRAIN_DIR_PATH = '/home/user/vilin/MOT17/FRCNN/train'
TEST_DIR_PATH = '/home/user/vilin/MOT17/FRCNN/test'
IMG_HEIGHT = 227
IMG_WIDTH = 227


def test(solver):
    accuracy = 0
    batch_size = solver.test_nets[0].blobs['data'].num
    test_iters = int(len(Xt) / batch_size)
    for i in range(test_iters):
        solver.test_nets[0].forward()
        accuracy += solver.test_nets[0].blobs['accuracy'].data
    accuracy /= test_iters

    print("Accuracy: {:.3f}".format(accuracy))


def train():
    q = 1


def img_to_net_input(img):
    input = np.moveaxis(img, 2, 0)
    input = np.expand_dims(input, axis=0)

    return input


def get_kvp_from_file(filepath):
    kvp = {}
    with open(filepath) as file:
        for line in file:
            key, value = line.partition("=")[::2]
            kvp[key.strip()] = value

    return kvp


def get_sub_dirs(folder):
    return next(os.walk(folder))[1]


def get_batch(root_dataset_dir):
    subdirs = get_sub_dirs(root_dataset_dir)
    print(subdirs)
    for seq_dir_name in subdirs:
        seq_dir_path = os.path.join(root_dataset_dir, seq_dir_name)
        seq_info_filepath = os.path.join(seq_dir_path, 'seqinfo.ini')
        seq_info = get_kvp_from_file(seq_info_filepath)

        images_dir = os.path.join(seq_dir_path)
        y_filepath = os.path.join(seq_dir_path, 'gt', 'gt.txt')

        print(seq_info['imDir'])
        print(seq_info['imExt'])
        print(images_dir)


def main():
    caffe.set_mode_gpu()
    caffe.set_device(0)

    get_batch(TRAIN_DIR_PATH)

    img_1_origin = cv2.imread(IMG_1)
    img_2_origin = cv2.imread(IMG_2)

    img_1_resized = cv2.resize(img_1_origin, (IMG_HEIGHT, IMG_WIDTH))
    img_2_resized = cv2.resize(img_2_origin, (IMG_HEIGHT, IMG_WIDTH))

    bbox = [1, 2, 3, 4]

    net = caffe.Net(MODEL_FILE, caffe.TEST)

    image_input = img_to_net_input(img_1_resized)
    target_input = img_to_net_input(img_2_resized)
    bbox_input = np.expand_dims(np.expand_dims(bbox, axis=3), axis=4)

    net.blobs['image'].data[...] = image_input
    net.blobs['target'].data[...] = target_input
    net.blobs['bbox'].data[...] = bbox_input

    print(net.blobs['bbox'].data.shape)

    # print(net.blobs['out'].data)

    solver = caffe.SGDSolver(SOLVER_FILE)
    solver.step(1)
    # solver.net.backward()

    # print('origin img_1_origin.shape = {}'.format(img_1_origin.shape))
    # print('origin img_2_origin.shape = {}'.format(img_2_origin.shape))
    # print('origin img_1_resized.shape = {}'.format(img_1_resized.shape))
    # print('origin img_2_resized.shape = {}'.format(img_2_resized.shape))


if __name__ == '__main__':
    main()
