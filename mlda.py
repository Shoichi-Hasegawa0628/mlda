# -*- coding: utf-8 -*-
import numpy as np
import random
import math
import csv

from numpy.random import set_state
import pylab
import pickle
import os

# ハイパーパラメータ
__alpha = 1.0
__beta = 1.0
epoch_num = 100

 # 学習エポック

def plot( n_dz, liks, D, K ):
    #print ("対数尤度", liks[-1])
    doc_dopics = np.argmax( n_dz , 1 )
    #print ("分類結果", doc_dopics)
    #print ("---------------------")


    # グラフ表示
    pylab.clf()
    pylab.subplot("121")
    pylab.title( "P(z|d)" )
    pylab.imshow( n_dz / np.tile(np.sum(n_dz,1).reshape(D,1),(1,K)) , interpolation="none" )
    pylab.subplot("122")
    pylab.title( "liklihood" )
    pylab.plot( range(len(liks)) , liks )
    pylab.draw()
    pylab.pause(0.1)

def calc_lda_param( docs_mdn, topics_mdn, K, dims ):
    M = len(docs_mdn)
    D = len(docs_mdn[0])

    # 各物体dにおいてトピックzが発生した回数
    n_dz = np.zeros((D,K))

    # 各トピックzにおいて特徴wが発生した回数
    n_mzw = [ np.zeros((K,dims[m])) for m in range(M)]

    # 各トピックが発生した回数
    n_mz = [ np.zeros(K) for m in range(M) ]

    # 数え上げ処理
    for d in range(D):
        for m in range(M):
            if dims[m]==0:
                continue
            N = len(docs_mdn[m][d])    # 物体に含まれる特徴数
            for n in range(N):
                w = docs_mdn[m][d][n]       # 物体dのn番目の特徴インデックス
                z = topics_mdn[m][d][n]     # 特徴に割り当てられているトピック
                n_dz[d][z] += 1
                n_mzw[m][z][w] += 1
                n_mz[m][z] += 1

    return n_dz, n_mzw, n_mz


def sample_topic( d, w, n_dz, n_zw, n_z, K, V ):
    P = [ 0.0 ] * K

    # 累積確率を計算
    P = (n_dz[d,:] + __alpha )*(n_zw[:,w] + __beta) / (n_z[:] + V *__beta)
    for z in range(1,K):
        P[z] = P[z] + P[z-1]

    # サンプリング
    rnd = P[K-1] * random.random()
    for z in range(K):
        if P[z] >= rnd:
            return z



# 単語を一列に並べたリスト変換
def conv_to_word_list( data ):
    V = len(data)
    doc = []
    for v in range(V):  # v:語彙のインデックス
        for n in range(data[v]): # 語彙の発生した回数分for文を回す
            doc.append(v)
    return doc

# 尤度計算関数
def calc_liklihood( data, n_dz, n_zw, n_z, K, V  ):
    lik = 0

    P_wz = (n_zw.T + __beta) / (n_z + V *__beta)
    for d in range(len(data)):
        Pz = (n_dz[d] + __alpha )/( np.sum(n_dz[d]) + K *__alpha )
        Pwz = Pz * P_wz
        Pw = np.sum( Pwz , 1 ) + 0.000001
        lik += np.sum( data[d] * np.log(Pw) )

    return lik


"""
def save_z(topics_mdn, M, D):
    # 単語ごとに割り当てられたトピックを保存 (物体の画像と単語に割り当てられたトピックを分けて保存)
    np.savetxt("z.csv", topics_mdn, delimiter=",", fmt="%s")

    S = 3
    # トピックの頻度情報を保存
    frequency_topic = [[ 0 for s in range(S) ] for d in range(D)]
    f = open('z_frequency.csv', 'w')
    writer = csv.writer(f)

    for m in range(M):
        for d in range(D):
            for n in range(len(topics_mdn[m][d])):
                if topics_mdn[m][d][n] == 0:
                    frequency_topic[d][0] += 1

                elif topics_mdn[m][d][n] == 1:
                    frequency_topic[d][1] += 1
                
                else:
                    frequency_topic[d][2] += 1
        
        writer.writerows(frequency_topic)
        frequency_topic = [[ 0 for s in range(S) ] for d in range(D)]


def save_z(topics_mdn, M, D):
    # 単語ごとに割り当てられたトピックを保存 (物体の画像と単語に割り当てられたトピックを一緒に保存)
    np.savetxt("z.csv", topics_mdn, delimiter=",", fmt="%s")

    S = 3
    # トピックの頻度情報を保存
    temporary_topic = [[ 0 for s in range(S) ] for d in range(D)]
    frequency_topic = [[ 0 for s in range(S) ] for d in range(D)]

    for m in range(M):
        for d in range(D):
            for n in range(len(topics_mdn[m][d])):
                if topics_mdn[m][d][n] == 0:
                    frequency_topic[d][0] += 1

                elif topics_mdn[m][d][n] == 1:
                    frequency_topic[d][1] += 1
                
                else:
                    frequency_topic[d][2] += 1
        
        if M == 0:
            #writer.writerows(frequency_topic)
            temporary_topic = frequency_topic
            frequency_topic = [[ 0 for s in range(S) ] for d in range(D)]
        
        # 画像と単語のトピック頻度数を合わせる
        else:
            for d in range(D):
                for s in range(S):
                    frequency_topic[d][s] += temporary_topic[d][s]
                    print("保存されるデータ:", frequency_topic[d][s])
                    
            f = open('z_frequency.csv', 'w')
            writer = csv.writer(f)
            writer.writerows(frequency_topic)
"""

def save_model( save_dir, n_dz, n_mzw, n_mz, M, dims ):
    try:
        os.mkdir( save_dir )
    except:
        pass

    Pdz = n_dz + __alpha
    Pdz = (Pdz.T / Pdz.sum(1)).T
    np.savetxt( os.path.join( save_dir, "Pdz.txt" ), Pdz, fmt=str("%f") ) #これがθ_dkに相当する？

    for m in range(M):
        Pwz = (n_mzw[m].T + __beta) / (n_mz[m] + dims[m] *__beta)
        Pdw = Pdz.dot(Pwz.T)
        np.savetxt( os.path.join( save_dir, "Pmdw[%d].txt" % m ) , Pdw ) #これがφでは？

    with open( os.path.join( save_dir, "model.pickle" ), "wb" ) as f:
        pickle.dump( [n_mzw, n_mz], f )


def load_model( load_dir ):
    model_path = os.path.join( load_dir, "model.pickle" )
    with open(model_path, "rb" ) as f:
        a,b = pickle.load( f )

    return a,b

# MLDAのメイン処理
def mlda( data, K, num_itr=epoch_num, save_dir="model", load_dir=None ):
    pylab.ion()

    # 尤度のリスト
    liks = []

    M = len(data)       # モダリティの数, dataの要素数からモダリティを取得. 今回は2個
    #print("モダリティM = ", M)
    #print("モダリティM = ", type(M))

    dims = []
    for m in range(M):
        if data[m] is not None: # 中身があるかを確認
            dims.append( len(data[m][0]) ) # BoW, BoFの一行目のデータの要素数を取得して、特徴数を得る、今回は50
            #print(data[m])
            #print(data[m][0])
            #print("dim = ", len(data[m][0]))
            D = len(data[m])    # 物体の数 (データの行の数、今回は6)
            #print("D = ", D)
        else:
            dims.append( 0 )

    # data内の単語を一列に並べる（計算しやすくするため）
    docs_mdn = [[ None for i in range(D) ] for m in range(M)]
    topics_mdn = [[ None for i in range(D) ] for m in range(M)]
    for d in range(D):
         for m in range(M):
            if data[m] is not None:
                docs_mdn[m][d] = conv_to_word_list( data[m][d] )
                topics_mdn[m][d] = np.random.randint( 0, K, len(docs_mdn[m][d]) ) # 各単語ごとにランダムでトピックを割り当てる (最大3の値(0, 1, 2)で50個の単語に)
                #print("Topic =", topics_mdn[m][d])


    # LDAのパラメータを計算, ここから見てみる
    """
    n_dz        各物体dにおいてトピックzが発生した回数
    n_mzw       各トピックzにおいて特徴wが発生した回数
    n_mz        各トピックが発生した回数
    """
    n_dz, n_mzw, n_mz = calc_lda_param( docs_mdn, topics_mdn, K, dims ) #

    # 認識モードでは学習したパラメータを読み込む
    if load_dir:
        n_mzw, n_mz = load_model( load_dir )

    for it in range(num_itr): #ギブスサンプリング
        # メイン処理
        for d in range(D):
            for m in range(M):
                if data[m] is None:
                    continue

                N = len(docs_mdn[m][d]) # 物体dのモダリティmに含まれる特徴数
                #print("N: ", N)
                for n in range(N):
                    #print("M, D, N:",M, D, N)
                    w = docs_mdn[m][d][n]       # 特徴のインデックス
                    z = topics_mdn[m][d][n]     # 特徴に割り当てられているカテゴリ


                    # データを取り除きパラメータの更新
                    n_dz[d][z] -= 1

                    if not load_dir:
                        n_mzw[m][z][w] -= 1
                        n_mz[m][z] -= 1

                    # サンプリング
                    z = sample_topic( d, w, n_dz, n_mzw[m], n_mz[m], K, dims[m] )

                    # データをサンプリングされたクラスに追加してパラメータを更新
                    topics_mdn[m][d][n] = z
                    #if d == D-1 and m == M-1 and n == N-1:
                        #save_z(topics_mdn, M, D) 
                        #print("object_length: ", len(topics_mdn))

                    n_dz[d][z] += 1

                    if not load_dir:
                        n_mzw[m][z][w] += 1
                        n_mz[m][z] += 1

        lik = 0
        for m in range(M):
            if data[m] is not None:
                lik += calc_liklihood( data[m], n_dz, n_mzw[m], n_mz[m], K, dims[m] )
        liks.append( lik )
        plot( n_dz, liks, D, K )

    #print("n_dz = ", n_dz)
    #print("n_mzw = ", n_mzw)
    #print("n_mz = ", n_mz)

    save_model( save_dir, n_dz, n_mzw, n_mz, M, dims )

    pylab.ioff()
    pylab.show()

def main():

#Bow, BoF表現したdataを格納, data = [[wordのdata], [visualのデータ]]

    topic = 3
    data = []
    data.append( np.loadtxt( "./bof/histgram_v.txt" , dtype=np.int32) )
    data.append( np.loadtxt( "./bow/histgram_w.txt" , dtype=np.int32)*5 )
    #print("data = ",data)
    mlda( data, topic, 100, "learn_result" )

    #data[1] = None
    #mlda( data, topic, 10, "recog_result" , "learn_result" )


if __name__ == '__main__':
    main()
