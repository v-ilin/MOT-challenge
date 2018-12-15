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


def get_batch(root_dataset_dir):
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

                # img = cv2.resize(img, (IMG_HEIGHT, IMG_WIDTH))
                # target_img = cv2.resize(target_img, (IMG_HEIGHT, IMG_WIDTH))

                img_bbox_left_bottom, img_bbox_right_top = get_bbox_coord(row)
                target_img_bbox_left_bottom, target_img_bbox_right_top = get_bbox_coord(target_row)

                # return (img, img_bbox), (target_img, target_img_bbox)

                cv2.rectangle(img, img_bbox_left_bottom, img_bbox_right_top, (255,0,0), 2)
                cv2.imwrite('img_with_bbox.jpg', img)

                cv2.rectangle(target_img, target_img_bbox_left_bottom, target_img_bbox_right_top, (255,0,0), 2)
                cv2.imwrite('target_img_with_bbox.jpg', target_img)

                print('img_filepath = {}'.format(img_filepath))
                print('target_img_filepath = {}'.format(target_img_filepath))
                print('img.shape = {}'.format(img.shape))
                print('target_img.shape = {}'.format(target_img.shape))
                break
            break

            print('object {}, frame count = {}'.format(name, group.shape[0]))
            # print(group)
        # print(y_df_grouped)
        break


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
