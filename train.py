import os
import datetime

import numpy as np
import cv2
import pandas as pd
import caffe

from common import utils

# os.environ['GLOG_minloglevel'] = '2'

MODEL_FILE = 'train.prototxt'
SOLVER_FILE = 'solver.prototxt'
OUTPUT_DIR = 'output'
TRAIN_DIR_PATH = '/home/user/vilin/MOT17/FRCNN/train'
TEST_DIR_PATH = '/home/user/vilin/MOT17/FRCNN/test'
IMG_HEIGHT = 227
IMG_WIDTH = 227


def get_kvp_from_file(filepath):
    kvp = {}
    with open(filepath) as file:
        for line in file:
            line = line.replace('\r\n', '')
            line = line.replace('\n', '')

            key, value = line.partition("=")[::2]
            kvp[key.strip()] = value

    return kvp


def get_sub_dirs(folder):
    return next(os.walk(folder))[1]


def compose_img_filepath(dir, img_id, img_ext):
    img_id_str = str(int(img_id)).zfill(6)
    img_filename = img_id_str + img_ext
    img_filepath = os.path.join(dir, img_filename)

    return img_filepath


def get_bbox_coord(csv_row):
    x1 = int(csv_row[2])
    y1 = int(csv_row[3])

    width = int(csv_row[4])
    height = int(csv_row[5])

    x2 = x1 + width
    y2 = y1 + height

    return (x1, y1), (x2, y2)


def scale_bbox_coord(img, point1, point2, target_width, target_height):
    x_scale = target_width / float(img.shape[1])
    y_scale = target_height / float(img.shape[0])

    (x1, y1) = point1
    (x2, y2) = point2

    x1_scaled = x1 * x_scale
    y1_scaled = y1 * y_scale

    x2_scaled = x2 * x_scale
    y2_scaled = y2 * y_scale

    # print('Origin bbox coord: x1 = {}, y1 = {}'.format(x1, y1))
    # print('Scaled bbox coord: x1 = {}, y1 = {}. x_scale = {}, y_scale = {}'.format(x1_scaled, y1_scaled, x_scale, y_scale))
    # print('Origin bbox coord: x2 = {}, y2 = {}'.format(x2, y2))
    # print('Scaled bbox coord: x2 = {}, y2 = {}. x_scale = {}, y_scale = {}'.format(x2_scaled, y2_scaled, x_scale, y_scale))

    return (int(x1_scaled), int(y1_scaled)), (int(x2_scaled), int(y2_scaled))


def center_image_over_bbox(img, bbox):
    cv2.imwrite('center_image_over_bbox_origin.jpg', img)

    (x1, y1), (x2, y2) = bbox

    x_bbox_center = (x1 + x2) / 2.0
    x_bbox_center = int(x_bbox_center)

    y_bbox_center = (y1 + y2) / 2.0
    y_bbox_center = int(y_bbox_center)

    x_img_center = img.shape[1] / 2.0
    x_img_center = int(x_img_center)

    y_img_center = img.shape[0] / 2.0
    y_img_center = int(y_img_center)

    top = 0
    bottom = 0
    left = 0
    right = 0

    x_diff = img.shape[1] - x_bbox_center
    y_diff = img.shape[0] - y_bbox_center

    x_already_filled = img.shape[1] - x_diff
    y_already_filled = img.shape[0] - y_diff

    # print('x_img_center = {}, x_bbox_center = {}, x_diff = {}'.format(x_img_center, x_bbox_center, x_diff))

    if x_bbox_center < x_img_center:
        left = abs(x_diff - x_already_filled)
        left = int(left / 2.0)
    elif x_bbox_center > x_img_center:
        right = abs(x_diff - x_already_filled)
        right = int(right / 2.0)

    if y_bbox_center < y_img_center:
        top = abs(y_diff - y_already_filled)
        top = int(top / 2.0)
    elif y_bbox_center < y_img_center:
        bottom = abs(y_diff - y_already_filled)
        bottom = int(bottom / 2.0)

    # print('origin img.shape = {}'.format(img.shape))
    if left != 0:
        img = img[:, :-left]
        # print('img - left = {}'.format(img.shape))

    if right != 0:
        img = img[:, right:]
        # print('img - left = {}'.format(img.shape))

    if bottom != 0:
        img = img[:-bottom, :]
        # print('img - bottom = {}'.format(img.shape))

    if top != 0:
        img = img[top:, :]
        # print('img - top = {}'.format(img.shape))

    result = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, 0)

    # print('img + padding = {}'.format(result.shape))
    # cv2.rectangle(result, (x1, y1), (x2, y2), (255, 0, 0), 16)
    # cv2.circle(result, (x_bbox_center, y_bbox_center), 6, (255, 0, 0), 16)
    # cv2.circle(result, (x_img_center, y_img_center), 6, (0, 0, 255), 16)
    # cv2.line(result, (x_bbox_center, y_bbox_center), (x_img_center, y_img_center),(0, 255, 0), 4)

    cv2.imwrite('img_centered_over_bbox.jpg', result)

    return result


def get_next_track_pair(root_dataset_dir):
    subdirs = get_sub_dirs(root_dataset_dir)

    for seq_dir_name in subdirs:
        seq_dir_path = os.path.join(root_dataset_dir, seq_dir_name)
        seq_info_filepath = os.path.join(seq_dir_path, 'seqinfo.ini')
        seq_info = get_kvp_from_file(seq_info_filepath)

        images_dir = os.path.join(seq_dir_path, seq_info['imDir'])
        y_filepath = os.path.join(seq_dir_path, 'gt', 'gt.txt')
        y_df = pd.read_csv(y_filepath, header=None)

        y_df_grouped = y_df.groupby(1)

        for name, group in y_df_grouped:
            for index, row in group.iterrows():
                target_row_index = index + 1

                if target_row_index >= group.shape[0]:
                    continue

                target_row = group.take([target_row_index])

                img_filepath = compose_img_filepath(images_dir, row[0], seq_info['imExt'])
                target_img_filepath = compose_img_filepath(images_dir, target_row[0], seq_info['imExt'])

                img = cv2.imread(img_filepath)
                target_img = cv2.imread(target_img_filepath)

                img_bbox_point1, img_bbox_point2 = get_bbox_coord(row)
                target_img_bbox_point1, target_img_bbox_point2 = get_bbox_coord(target_row)

                img_centered = center_image_over_bbox(img, (img_bbox_point1, img_bbox_point2))

                target_img_bbox_point1, target_img_bbox_point2 = scale_bbox_coord(target_img, target_img_bbox_point1, target_img_bbox_point2, IMG_WIDTH, IMG_HEIGHT)

                img_centered = cv2.resize(img_centered, (IMG_WIDTH, IMG_HEIGHT))
                target_img = cv2.resize(target_img, (IMG_WIDTH, IMG_HEIGHT))

                target_img_bbox = (target_img_bbox_point1, target_img_bbox_point2)

                yield img_centered, target_img, target_img_bbox

            # print('object {}, frame count = {}'.format(name, group.shape[0]))


def main():
    caffe.set_mode_gpu()
    caffe.set_device(0)

    net = caffe.Net(MODEL_FILE, caffe.TRAIN)
    solver = caffe.SGDSolver(SOLVER_FILE)

    for img, target_img, target_img_bbox_points in get_next_track_pair(TRAIN_DIR_PATH):
        (x1_target, y1_target), (x2_target, y2_target) = target_img_bbox_points

        target_img_bbox = [x1_target, y1_target, x2_target, y2_target]

        img_input = utils.img_to_net_input(img)
        target_input = utils.img_to_net_input(target_img)

        target_bbox_input = np.expand_dims(np.expand_dims(target_img_bbox, axis=3), axis=4)

        net.blobs['image'].data[...] = img_input
        net.blobs['target'].data[...] = target_input
        net.blobs['bbox'].data[...] = target_bbox_input

        solver.step(1)
        break

    # model_filename = 'final_model_{}.caffemodel'.format(datetime.datetime.now().isoformat())
    # model_filepath = os.path.join(OUTPUT_DIR, model_filename)
    # net.save(model_filepath)


if __name__ == '__main__':
    main()
