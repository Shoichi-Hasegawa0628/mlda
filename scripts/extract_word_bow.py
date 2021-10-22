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
        self.object_nums = 0


    def word_server(self, yolov3_image, status, observed_img_idx, count):
        if status == "learn":
            for c in range(len(OBJECT_CLASS)):
                for n in range(len(OBJECT_NAME[c])):
                    self.separate_word(OBJECT_CLASS[c], OBJECT_NAME[c][n])
                    self.make_dic(status)
                self.object_nums += len(OBJECT_NAME[c])
            print(len(self.lines))
            print(len(self.dictionary))
            print(self.lines)
            self.hist_w = np.zeros((self.object_nums, len(self.dictionary)))
            ct = 0
            for i in range(len(self.lines)):
                self.make_word_bow(ct, status, self.lines[i])
                if (i + 1) % (len(self.lines) / self.object_nums) == 0:
                    ct += 1

        else:
            result = self.estimate(yolov3_image, status, observed_img_idx, count)
            return result


    def separate_word(self, object_class, object_name):
        for line in codecs.open(LEARNING_DATASET_FOLDER + "/" +
                                object_class + "/" +
                                object_name + "/" +
                                TEACHING_DATA, "r", "utf-8").readlines():
            line = line.rstrip("\r\n")
            words = line.split(" ")
            self.lines.append(words)
        rospy.loginfo("Finished separating words\n")


    def make_dic(self, status):
        for line in self.lines:
            for word in line:
                if word not in self.dictionary:
                    self.dictionary.append(word)

        codecs.open(PROCESSING_DATA_FOLDER + "/" +
                    "bow" + "/" + status + "/" +
                    WORD_DICTIONARY, "w", "utf-8").write("\n".join(self.dictionary))
        rospy.loginfo("Saved the word dictionary as %s\n", WORD_DICTIONARY)


    def make_word_bow(self, ct, status, lines):
        for i, word in enumerate(lines):
            #print(word)
            idx = self.dictionary.index(word)
            self.hist_w[ct, idx] += 1

        np.savetxt(PROCESSING_DATA_FOLDER + "/" +
                   "bow" + "/" + status + "/" +
                   WORD_HIST, self.hist_w, fmt=str("%d"))
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
        word = np.loadtxt(ESTIMATE_RESULT_FOLDER + "/{}/Pmdw[1]_{}.txt".format(observed_img_idx, count))
        word_file = open(PROCESSING_DATA_FOLDER + "/bow/learn/" + WORD_DICTIONARY)
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
        with open(ESTIMATE_RESULT_FOLDER + "/{}/estimate_word_{}.txt".format(observed_img_idx, count), mode='w') as f:

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

        if os.path.exists(ESTIMATE_RESULT_FOLDER + "/{}".format(observed_img_idx)) is True:
            pass
        else:
            os.mkdir(ESTIMATE_RESULT_FOLDER + "/{}".format(observed_img_idx))

        file = pathlib.Path(ESTIMATE_RESULT_FOLDER + "/{}/estimate_histgram_word_{}.txt".format(observed_img_idx, count))
        file.touch()
        np.savetxt(ESTIMATE_RESULT_FOLDER + "/{}/estimate_histgram_word_{}.txt".format(observed_img_idx, count),
                   estimate_hist.reshape(1, -1),
                   fmt=str("%d"))

        return estimate_result

if __name__ == '__main__':
    extract_word_bow = ExtractWordBow()

    #単独で実行させたいとき
    #status = "learn"
    #extract_word_bow.word_server(None, status, None, None)
