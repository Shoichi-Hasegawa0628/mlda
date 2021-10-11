#! /usr/bin/env python
# -*- coding: utf-8 -*-
# 必要なライブラリ
# pip install pathlib

# Standard Library
from __future__ import unicode_literals
import codecs
import pathlib

# Third Party
import numpy as np
from std_msgs.msg import String
import rospy

# Self-made Modules
from __init__ import *


class ExtractWordBow():

    def __init__(self):
        rospy.loginfo("Ready em_mlda/word")
        self.lines = []
        self.dictionary = []
        self.hist_w = np.zeros(3)


    def word_server(self, yolov3_image, status, observed_img_idx, count):
        if status == "learn":
            ct = 1
            for c in range(len(OBJECT_CLASS)):
                for n in range(len(OBJECT_NAME[c])):
                    self.separate_word(OBJECT_CLASS[c], OBJECT_NAME[c][n])
                    self.make_dic()

                    if c == 0 & n == 0:
                        self.hist_w = np.zeros((len(self.lines), len(self.dictionary)))

                    self.make_word_bow(ct)
                    self.lines = []
                    ct += 1

        elif status == "estimate":
            pass


    def separate_word(self, object_class, object_name):
        for line in codecs.open(LEARNING_DATASET_FOLDER + "/" +
                                object_class + "/" +
                                object_name + "/" +
                                TEACHING_DATA, "r", "utf-8").readlines():
            line = line.rstrip("\r\n")
            words = line.split(" ")
            self.lines.append(words)
        rospy.loginfo("Finished separating words\n")


    def make_dic(self):
        for line in self.lines:
            for word in line:
                if word not in self.dictionary:
                    self.dictionary.append(word)

        codecs.open(DATA_FOLDER + "/" + WORD_DICTIONARY, "w", "utf-8").write("\n".join(self.dictionary))
        rospy.loginfo("Saved the word dictionary as %s\n", WORD_DICTIONARY)


    def make_word_bow(self, ct):
        for words in enumerate(self.lines):
            for word in words:
                idx = self.dictionary.index(word)
                self.hist_w[ct, idx] += 1

        np.savetxt(DATA_FOLDER + "/" + WORD_HIST, self.hist_w, fmt=str("%d"))
        rospy.loginfo("Saved the word histgram as %s\n", WORD_HIST)


    def add_word(self, object_class, object_name, add_word):
        with open(LEARNING_DATASET_FOLDER + "/" +
                  object_class + "/" +
                  object_name + "/" +
                  TEACHING_DATA) as f:
            current_words = [s.strip() for s in f.readlines()]
        current_words.append(add_word)

        with open(LEARNING_DATASET_FOLDER + "/" +
                  object_class + "/" +
                  object_name + "/" +
                  TEACHING_DATA, mode='w') as f:
            f.write('\n'.join(current_words))
        rospy.loginfo("Added the new word %s to object %d", add_word, object_name)


    def estimate(self, yolov3_image, status, observed_img_idx, count):
        word = np.loadtxt(PROCESSING_DATA_FOLDER + "/" +
                          "bow/{}/{}/Pmdw[1]_{}.txt".format(status, observed_img_idx, count))
        word_file = open(DATA_FOLDER + "/" + WORD_DICTIONARY)
        word_dic = word_file.readlines()
        word_dic = [convert.replace("\n", "") for convert in word_dic]
        word_dic = [convert.replace(".", "") for convert in word_dic]
        # print(word_dic)
        unsorted_max_indices = np.argpartition(-word, 10)[:10]  # 上位10つの単語を表示
        # print(unsorted_max_indices)
        # print(len(unsorted_max_indices))
        # print(len(word))
        # print(word)
        y = word[unsorted_max_indices]
        indices = np.argsort(-y)
        max_k_indices = unsorted_max_indices[indices]
        rospy.loginfo("Estimate the word as")
        index_counter = 0
        estimated_words = "I estimated this object. "
        estimate_result = {}
        with open(PROCESSING_DATA_FOLDER + "/" +
                  "bow/{}/{}/estimate_word_{}.txt".format(status, observed_img_idx, count), mode='w') as f:

            for word_index in max_k_indices:
                # print(word_dic[word_index])
                # estimated_words += word_dic[word_index] + " " + str(round(y[index_counter] * 100, 2)) + "%. "
                estimated_words += "No." + \
                                   str(index_counter + 1) + ". " + \
                                   word_dic[word_index] + " " + \
                                   str(round(y[index_counter] * 100, 1)) + "%. "
                # estimated_words += word_dic[word_index] + "."
                f.write(word_dic[word_index] + " " + str(round(y[index_counter] * 100, 1)) + "%" + "\n")
                estimate_result["{}{}_".format(observed_img_idx, count) + word_dic[word_index]] = round(y[index_counter] * 100, 1)
                index_counter += 1

        estimate_hist = np.zeros(len(word_dic))
        for word_index in max_k_indices:
            estimate_hist[word_index] += 1

        if os.path.exists(PROCESSING_DATA_FOLDER + "/" + "bow/{}/{}".format(status, observed_img_idx)) is True:
            pass
        else:
            os.mkdir(PROCESSING_DATA_FOLDER + "/" + "bow/{}/{}".format(status, observed_img_idx))

        file = pathlib.Path(PROCESSING_DATA_FOLDER + "/" +
                            "bow/{}/{}/estimate_histgram_word_{}.txt".format(status, observed_img_idx, count))
        file.touch()
        np.savetxt(PROCESSING_DATA_FOLDER + "/" +
                   "bow/{}/{}/estimate_histgram_word_{}.txt".format(status, observed_img_idx, count),
                   estimate_hist.reshape(1, -1), fmt=str("%d"))
        return estimate_result


if __name__ == '__main__':
    pass
