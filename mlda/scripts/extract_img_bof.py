#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Standard Library
import cv2
from cv_bridge import CvBridge, CvBridgeError
import glob
import pathlib

# Third Party
import numpy as np
import rospy
from sensor_msgs.msg import Image

# Self-made Modules
from __init__ import *


class ExtractImgBof():

    def __init__(self):
        self.knn = cv2.ml.KNearest_create()
        #self.detector = cv2.AKAZE_create()
        self.detector = cv2.ORB_create()
        self.cut_img = 0
        self.hists = []
        rospy.loginfo("Ready ...\n")

    def img_server(self, status, count, observed_img_idx, yolov3_image, mode):
        if status == "learn":
            ct = 1
            for c in range(len(OBJECT_CLASS)):
                for n in range(len(OBJECT_NAME[c])):
                    self.make_codebook(OBJECT_CLASS[c], OBJECT_NAME[c][n], status, None, None, mode)
                    self.make_img_bof(status, OBJECT_CLASS[c], OBJECT_NAME[c][n], None, None, mode)
                    ct += 1

        else:
            print(type(yolov3_image))
            self.cut_img = self.image_callback(yolov3_image)
            #self.cut_img = yolov3_image
            print(type(self.cut_img))
            result = self.make_codebook(None, None, status, count, observed_img_idx, mode)
            if result == 0:
                return 0
            self.make_img_bof(status, None, None, count, observed_img_idx, mode)

    def calc_feature(self, img):
        kp, discriptor = self.detector.detectAndCompute(img, None)

        """
        if discriptor is None:
            print("特徴量抽出に失敗しました。\n")
            return 0, 0

        elif len(discriptor) < 50:
            print("指定したcode_book_sizeよりも特徴数が少ないため失敗しました。\n")
            return 0, np.array(discriptor, dtype=np.float32)
        """

        return 1, np.array(discriptor, dtype=np.float32)

    def make_codebook(self, object_class, object_name, status, count, observed_img_idx, mode):
        bow_trainer = cv2.BOWKMeansTrainer(CODE_BOOK_SIZE)
        if status == "learn":
            image_files = glob.glob(LEARNING_DATASET_FOLDER + "/" +
                                    object_class + "/" +
                                    object_name + "/" +
                                    "image/*.png")
            for img in image_files:
                img = cv2.imread(img)
                result, f = self.calc_feature(img)
                bow_trainer.add(f)

            code_book = bow_trainer.cluster()
            np.savetxt(PROCESSING_DATA_FOLDER + "/" +
                       "bof/{}".format(status) + "/" +
                       CODE_BOOK, code_book)

        else:
            if mode == "0":
                result, f = self.calc_feature(self.cut_img)
                if result == 0:
                    return 0
                bow_trainer.add(f)
                code_book = bow_trainer.cluster()
                if os.path.exists(PROCESSING_DATA_FOLDER + "/" +
                                  "bof/{}/{}/{}".format(status, mode, observed_img_idx)
                                  ) is True:
                    pass

                else:
                    os.mkdir(PROCESSING_DATA_FOLDER + "/" + "bof/{}/{}/{}".format(status, mode, observed_img_idx))

                file = pathlib.Path(PROCESSING_DATA_FOLDER + "/" +
                                    "bof/{}/{}/{}".format(status, mode, observed_img_idx) + "/" +
                                    "codebook_{}.txt".format(count))
                file.touch()
                np.savetxt(PROCESSING_DATA_FOLDER + "/" +
                           "bof/{}/{}/{}".format(status, mode, observed_img_idx) + "/" +
                           "codebook_{}.txt".format(count),
                           code_book)

            else:
                result, f = self.calc_feature(self.cut_img)
                if result == 0:
                    return 0
                bow_trainer.add(f)
                code_book = bow_trainer.cluster()
                if os.path.exists(PROCESSING_DATA_FOLDER + "/" +
                                  "bof/{}/{}".format(status, observed_img_idx)
                                  ) is True:
                    pass

                else:
                    os.mkdir(PROCESSING_DATA_FOLDER + "/" + "bof/{}/{}".format(status, observed_img_idx))

                file = pathlib.Path(PROCESSING_DATA_FOLDER + "/" +
                                    "bof/{}/{}".format(status, observed_img_idx) + "/" +
                                    "codebook_{}.txt".format(count))
                file.touch()
                np.savetxt(PROCESSING_DATA_FOLDER + "/" +
                           "bof/{}/{}".format(status, observed_img_idx) + "/" +
                           "codebook_{}.txt".format(count),
                           code_book)

    def make_img_bof(self, status, object_class, object_name, count, observed_img_idx, mode):
        if status == "learn":
            code_book = np.loadtxt(PROCESSING_DATA_FOLDER + "/" +
                                   "bof/{}".format(status) + "/" +
                                   CODE_BOOK, dtype=np.float32)
            knn = cv2.ml.KNearest_create()
            knn.train(code_book, cv2.ml.ROW_SAMPLE, np.arange(len(code_book), dtype=np.float32))
            h = np.zeros(len(code_book))  # 要素数が50で要素を0とする1次元配列の作成
            image_files = glob.glob(LEARNING_DATASET_FOLDER + "/" +
                                    object_class + "/" +
                                    object_name + "/" +
                                    "image/*.png")

            for img in image_files:
                img = cv2.imread(img)
                result, f = self.calc_feature(img)  # 特徴量計算
                idx = knn.findNearest(f, 1)[1]  # K=1で分類

                for i in idx:  # 頻度をカウントしている, idexの値自体が1〜50の値を取るから、それでヒットした回数をカウントしていく。
                    h[int(i)] += 1

            self.hists.append(h)
            np.savetxt(PROCESSING_DATA_FOLDER + "/" +
                       "bof/{}".format(status) + "/" +
                       IMG_HIST, self.hists, fmt=str("%d"))

        else:
            if mode == "0":
                code_book = np.loadtxt(PROCESSING_DATA_FOLDER + "/" +
                                        "bof/{}/{}/{}".format(status, mode, observed_img_idx) + "/" +
                                        "codebook_{}.txt".format(count),
                                        dtype=np.float32)
                self.knn.train(code_book, cv2.ml.ROW_SAMPLE, np.arange(len(code_book), dtype=np.float32))
                h = np.zeros(len(code_book))
                result, f = self.calc_feature(self.cut_img)
                print("N: {}".format(f))
                idx = self.knn.findNearest(f, 1)[1]

                for i in idx:
                    h[int(i)] += 1

                self.hists.append(h)
                if os.path.exists(PROCESSING_DATA_FOLDER + "/" +
                                  "bof/{}/{}/{}".format(status, mode, observed_img_idx)
                                  ) is True:
                    pass

                else: ##修正必要！！！
                    os.mkdir(PROCESSING_DATA_FOLDER + "/" +
                             "bof/{}/{}/{}".format(status, mode, observed_img_idx))

                file = pathlib.Path(PROCESSING_DATA_FOLDER + "/" +
                                    "bof/{}/{}/{}".format(status, mode, observed_img_idx) + "/" +
                                    "histgram_img_{}.txt".format(count))
                file.touch()
                np.savetxt(PROCESSING_DATA_FOLDER + "/" +
                           "bof/{}/{}/{}".format(status, mode, observed_img_idx) + "/" +
                           "histgram_img_{}.txt".format(count),
                           self.hists,
                           fmt=str("%d"))

            else:
                result, f = self.calc_feature(self.cut_img)
                idx = self.knn.findNearest(f, 1)[1]
                code_book = np.loadtxtx(PROCESSING_DATA_FOLDER + "/" +
                                        "bof/{}/{}".format(status, observed_img_idx) + "/" +
                                        "codebook_{}.txt".format(count),
                                        dtype=np.float32)
                h = np.zeros(len(code_book))
                for i in idx:
                    h[int(i)] += 1

                self.hists.append(h)
                if os.path.exists(PROCESSING_DATA_FOLDER + "/" +
                                  "bof/{}/{}".format(status, observed_img_idx)
                                  ) is True:
                    pass

                else:
                    os.mkdir(PROCESSING_DATA_FOLDER + "/" +
                             "bof/{}/{}".format(status, observed_img_idx))

                file = pathlib.Path(PROCESSING_DATA_FOLDER + "/" +
                                    "bof/{}/{}".format(status, observed_img_idx) + "/" +
                                    "histgram_img_{}.txt".format(count))
                file.touch()
                np.savetxt(PROCESSING_DATA_FOLDER + "/" +
                           "bof/{}/{}".format(status, observed_img_idx) + "/" +
                           "histgram_img_{}.txt".format(count),
                           self.hists,
                           fmt=str("%d"))

    def add_img(self, object_class, object_name, add_img):
        image_files = glob.glob(LEARNING_DATASET_FOLDER + "/" +
                                object_class + "/" +
                                object_name + "/" +
                                "image/*.png")
        cv2.imwrite(LEARNING_DATASET_FOLDER + "/" +
                    object_class + "/" +
                    object_name + "/" +
                    "image/{}.png".format(len(image_files) + 1), add_img)

    def image_callback(self, image):
        bridge = CvBridge()
        try:
            img = bridge.imgmsg_to_cv2(image, 'passthrough')
        except CvBridgeError as e:
            print(e)
        return img


if __name__ == '__main__':
    extract_img_bof = ExtractImgBof()

    # 単独で実行させたいとき
    status = "learn"
    extract_img_bof.img_server(status, None, None, None)
