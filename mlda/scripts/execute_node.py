#! /usr/bin/env python
# -*- coding: utf-8 -*-
# apt-get install python-tk

# Standard Library
import random
import os
import math
import csv
import sys
import codecs

# Third Party
import numpy as np
from numpy.random import set_state
import pylab
import pickle
import rospy
import roslib.packages

# Self-made Modules
from __init__ import *


class MLDA():

    def load_model(self, estimate_mode):
        with open(LEARN_RESULT_FOLDER + "/model.pickle", "rb") as f:
            a, b = pickle.load(f)
        return a, b

    def save_model(self, status, save_dir, n_dz, n_mzw, n_mz, docs_mdn, topics_mdn, M, dims, count):
        Pdz = n_dz + ALPHA
        Pdz = (Pdz.T / Pdz.sum(1)).T
        if status == "learn":
            np.savetxt(os.path.join(save_dir, "Pdz.txt"), Pdz, fmt=str("%f"))
        else:
            np.savetxt(os.path.join(save_dir, "Pdz_{}.txt".format(count)),
                       Pdz, delimiter=",", fmt=str("%f"))

        for m in range(M):
            Pwz = (n_mzw[m].T + BETA) / (n_mz[m] + dims[m] * BETA)
            Pdw = Pdz.dot(Pwz.T)
            if status == "learn":
                np.savetxt(os.path.join(save_dir, "Pmdw[%d].txt" % m), Pdw)
                np.savetxt(os.path.join(save_dir, "Pdw[%d].txt" % m), Pwz)
            else:
                np.savetxt(os.path.join(save_dir, "Pmdw[%d]_{}.txt".format(count) % m), Pdw, fmt=str("%f"))
                np.savetxt(os.path.join(save_dir, "Pdw[%d]_{}.txt".format(count) % m), Pwz, fmt=str("%f"))

        if status == "learn":
            with open(os.path.join(save_dir, "model.pickle"), "wb") as f:
                pickle.dump([n_mzw, n_mz], f)
        else:
            with open(os.path.join(save_dir, "model_{}.pickle".format(count)), "wb") as f:
                pickle.dump([n_dz, n_mzw, n_mz, docs_mdn, topics_mdn], f)

    def save_frequency_z(self, topics_mdn, Modality_num, Object_num):
        # 単語ごとに割り当てられた物体カテゴリを保存 (物体の画像と単語に割り当てられた物体カテゴリを一緒に保存)
        np.savetxt(str(roslib.packages.get_pkg_dir("spco2_mlda")) + "/data/pre_learning/co_frequency/co_word.csv", topics_mdn, delimiter=",", fmt="%s")

        # 物体ごとの物体カテゴリの割当回数を保存
        temporary_topic = [[0 for s in range(CATEGORYNUM)] for d in range(Object_num)]
        frequency_topic = [[0 for s in range(CATEGORYNUM)] for d in range(Object_num)]
        for m in range(Modality_num):
            if m == 1:
                continue
            for d in range(Object_num):
                for n in range(len(topics_mdn[m][d])):
                    if topics_mdn[m][d][n] == 0:
                        frequency_topic[d][0] += 1
                    elif topics_mdn[m][d][n] == 1:
                        frequency_topic[d][1] += 1

                    else:
                        frequency_topic[d][2] += 1

            if Modality_num == 0:
                # writer.writerows(frequency_topic)
                temporary_topic = frequency_topic
                frequency_topic = [[0 for s in range(CATEGORYNUM)] for d in range(Object_num)]

            # 画像と単語の物体カテゴリ割当て回数を合わせる
            else:
                for d in range(Object_num):
                    for s in range(CATEGORYNUM):
                        frequency_topic[d][s] += temporary_topic[d][s]
                        # print("保存されるデータ:", frequency_topic[d][s])

                f = open(str(roslib.packages.get_pkg_dir("spco2_mlda")) + "/data/pre_learning/co_frequency/co_frequency.csv", 'w')
                writer = csv.writer(f)
                writer.writerows(frequency_topic)

    def calc_liklihood(self, target_modality_num, n_dz, n_zw, n_z, V):
        lik = 0
        P_wz = (n_zw.T + BETA) / (n_z + V * BETA)

        if self.data[target_modality_num].ndim == 1:
            object_num = 1
        else:
            object_num = len(self.data[target_modality_num])

        for d in range(object_num):
            Pz = (n_dz[d] + ALPHA) / (np.sum(n_dz[d]) + CATEGORYNUM * ALPHA)
            Pwz = Pz * P_wz
            Pw = np.sum(Pwz, 1) + 0.000001
            lik += np.sum(self.data[target_modality_num][d] * np.log(Pw))

        return lik

    def conv_to_word_list(self, data):
        if data.ndim == 1:
            V = data.shape[0]
        else:
            V = len(data)

        doc = []
        for v in range(V):  # v:語彙のインデックス
            # 語彙の発生した回数文forを回す
            for n in range(data[v]):
                doc.append(v)
        return doc

    def sample_topic(self, target_object, Target_character_index, n_dz, n_zw, n_z, dimension_list):
        P = [0.0] * CATEGORYNUM

        # 累積確率を計算
        P = (n_dz[target_object, :] + ALPHA) * (n_zw[:,
                                                Target_character_index] + BETA) / (n_z[:] + dimension_list * BETA)
        for z in range(1, CATEGORYNUM):
            P[z] = P[z] + P[z - 1]

        # サンプリング
        rnd = P[CATEGORYNUM - 1] * random.random()
        for z in range(CATEGORYNUM):
            if P[z] >= rnd:
                return z

    def calc_lda_param(self, docs_mdn, topics_mdn, dims):
        Modality_num = len(docs_mdn)
        Object_num = len(docs_mdn[0])

        # 各物体dにおいてカテゴリzが発生した回数
        n_dz = np.zeros((Object_num, CATEGORYNUM))

        # 各カテゴリzにおいてモダリティーごとに特徴wが発生した回数
        n_mzw = [np.zeros((CATEGORYNUM, dims[m])) for m in range(Modality_num)]

        # モダリティーごとに各トピックが発生した回数
        n_mz = [np.zeros(CATEGORYNUM) for m in range(Modality_num)]

        # 数え上げる
        for d in range(Object_num):
            for m in range(Modality_num):
                if dims[m] == 0:
                    continue
                N = len(docs_mdn[m][d])  # 物体に含まれる特徴数
                for n in range(N):
                    w = docs_mdn[m][d][n]  # 物体dのn番目の特徴のインデックス
                    z = topics_mdn[m][d][n]  # 特徴に割り当てれれているトピック
                    n_dz[d][z] += 1
                    n_mzw[m][z][w] += 1
                    n_mz[m][z] += 1

        return n_dz, n_mzw, n_mz

    def plot(self, n_dz, liks, D):
        print("対数尤度：", liks[-1])
        doc_dopics = np.argmax(n_dz, 1)
        print("分類結果：", doc_dopics)
        print("---------------------")
        # グラフ表示
        pylab.clf()
        pylab.subplot("121")
        pylab.title("P(z|d)")
        pylab.imshow(n_dz / np.tile(np.sum(n_dz, 1).reshape(D, 1),
                                    (1, CATEGORYNUM)), interpolation="none")
        pylab.subplot("122")
        pylab.title("liklihood")
        pylab.plot(list(range(len(liks))), liks)
        pylab.draw()
        pylab.pause(0.01)

    def mlda_learn(self, status, count, mode, save_dir="model", estimate_mode=False):
        pylab.ion()
        liks = []
        Modality_num = len(self.data)
        dimension_list = []

        for Target_modality in range(Modality_num):
            if self.data[Target_modality] is not None:
                if self.data[Target_modality].ndim == 1:
                    dimension_list.append(self.data[Target_modality].shape[0])
                    Object_num = 1
                else:
                    # 次元数をdimension_listに追加
                    dimension_list.append(len(self.data[Target_modality][0]))
                    Object_num = len(self.data[Target_modality])  # 物体数
            else:
                dimension_list.append(0)

        # [NoneがObject_num個]がModality_num個
        docs_mdn = [[None for i in range(Object_num)]
                    for m in range(Modality_num)]
        topics_mdn = [[None for i in range(Object_num)]
                      for m in range(Modality_num)]

        # data内の単語を一列に並べる（計算しやすくするため）
        for Target_object in range(Object_num):
            for Target_modality in range(Modality_num):
                if self.data[Target_modality] is not None:
                    if Object_num == 1:
                        docs_mdn[Target_modality][Target_object] = self.conv_to_word_list(
                            self.data[Target_modality])
                        topics_mdn[Target_modality][Target_object] = np.random.randint(
                            0, CATEGORYNUM, len(docs_mdn[Target_modality][Target_object]))  # 各単語にランダムでトピックを割り当てる

                    else:
                        docs_mdn[Target_modality][Target_object] = self.conv_to_word_list(
                            self.data[Target_modality][Target_object])
                        topics_mdn[Target_modality][Target_object] = np.random.randint(
                            0, CATEGORYNUM, len(docs_mdn[Target_modality][Target_object]))  # 各単語にランダムでトピックを割り当てる

        # LDAのパラメータを計算
        n_dz, n_mzw, n_mz = self.calc_lda_param(docs_mdn, topics_mdn, dimension_list)

        # 推定モードでは学習済みのパラメータを読み込む
        if estimate_mode:
            n_mzw, n_mz = self.load_model(True)
        print("D: {}".format(Object_num), "M: {}".format(Modality_num))
        print(self.data)
        for It in range(ITERATION):
            # メインの処理
            for Target_object in range(Object_num):
                for Target_modality in range(Modality_num):
                    if self.data[Target_modality] is None:
                        continue

                    Target_character_num = len(docs_mdn[Target_modality][Target_object])  # 物体dのモダリティmに含まれる特徴数
                    print(Target_character_num)
                    for Target_character in range(Target_character_num):
                        # 特徴のインデックス
                        Target_character_index = docs_mdn[Target_modality][Target_object][Target_character]
                        # 特徴に割り当てられているカテゴリ
                        Target_character_category = topics_mdn[Target_modality][Target_object][Target_character]
                        # データを取り除きパラメータを更新
                        n_dz[Target_object][Target_character_category] -= 1

                        if not estimate_mode:
                            n_mzw[Target_modality][Target_character_category][Target_character_index] -= 1
                            n_mz[Target_modality][Target_character_category] -= 1

                        # サンプリング
                        Target_character_category = self.sample_topic(
                            Target_object, Target_character_index, n_dz, n_mzw[Target_modality],
                            n_mz[Target_modality], dimension_list[Target_modality])

                        # データをサンプリングされたクラスに追加してパラメータを更新
                        topics_mdn[Target_modality][Target_object][Target_character] = Target_character_category
                        if mode == "0":
                            if Target_modality == 0 and Target_character == Target_character_num - 1:
                                print("Save!")
                                self.save_frequency_z(topics_mdn, Modality_num, Object_num)
                            # print("object_length: ", len(topics_mdn)

                        n_dz[Target_object][Target_character_category] += 1

                        if not estimate_mode:
                            n_mzw[Target_modality][Target_character_category][Target_character_index] += 1
                            n_mz[Target_modality][Target_character_category] += 1

            lik = 0
            for Target_modality in range(Modality_num):
                if self.data[Target_modality] is not None:
                    lik += self.calc_liklihood(Target_modality, n_dz, n_mzw[Target_modality],
                                               n_mz[Target_modality], dimension_list[Target_modality])
            liks.append(lik)
            self.plot(n_dz, liks, Object_num)
            if It == ITERATION - 1:
                # if True:
                print("Iteration ", It + 1)
                pylab.close()
        if mode == "0":
            return
        self.save_model(status, save_dir, n_dz, n_mzw, n_mz, docs_mdn,
                        topics_mdn, Modality_num, dimension_list, count)
        pylab.ioff()
        pylab.show()

    def mlda_server(self, status, observed_img_idx, count, mode):
        if status == "learn":
            self.data = [np.loadtxt(PROCESSING_DATA_FOLDER + "/" +
                                    "bof" + "/" + status + "/" +
                                    IMG_HIST, dtype=np.int32),
                         np.loadtxt(PROCESSING_DATA_FOLDER + "/" +
                                    "bow" + "/" + status + "/" +
                                    WORD_HIST, dtype=np.int32) * 5]
            rospy.loginfo("Defalut learning mode start\n")
            self.mlda_learn(status, None, mode, LEARN_RESULT_FOLDER, False)

        else:
            if mode == "0":
                self.data = [np.loadtxt(PROCESSING_DATA_FOLDER + "/" +
                                        "bof/{}/{}/{}".format(status, mode,
                                                              observed_img_idx) + "/" + "histgram_img_{}.txt".format(
                    count),
                                        dtype=np.int32), None]
                self.mlda_learn(status, count, mode, ESTIMATE_RESULT_FOLDER + "/" +
                                "{}".format(observed_img_idx), True)
                return

            else:
                self.data = [np.loadtxt(PROCESSING_DATA_FOLDER + "/" +
                                        "bof/{}/{}/{}".format(status, mode,
                                                              observed_img_idx) + "/" + "histgram_img_{}.txt".format(
                    count),
                                        dtype=np.int32), None]
                self.mlda_learn(status, count, mode, ESTIMATE_RESULT_FOLDER + "/" +
                                "{}".format(observed_img_idx), True)

    def __init__(self):
        self.data = []
        #self.mlda_server("learn", None, None, None)


######################################################################################################

if __name__ == '__main__':
    # sys.stdout = codecs.getwriter('utf_8')(sys.stdout)
    # rospy.init_node('em_mlda_learn_server')
    MLDA()
    # rospy.spin()
