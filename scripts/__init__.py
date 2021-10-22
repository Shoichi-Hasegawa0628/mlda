#! /usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
import os
from geometry_msgs.msg import Pose
from geometry_msgs.msg import Point
import numpy as np
import random
import subprocess
import math

#from mlda_ros.srv import *
#from mlda_ros.msg import *

import sys
reload(sys)
sys.setdefaultencoding('utf-8')
import roslib.packages
import os

CODE_BOOK_SIZE = 50
ITERATION = 100
CATEGORYNUM = 3
ALPHA = 1.0
BETA = 1.0

LEARN_RESULT_FOLDER = "../data/learn_result"
ESTIMATE_RESULT_FOLDER = "../data/estimate_result"
PROCESSING_DATA_FOLDER = "../data/processing_data"
LEARNING_DATASET_FOLDER = str(roslib.packages.get_pkg_dir("mlda_dataset_original")) + "/rsj_exp"
OBJECT_CLASS = os.listdir(LEARNING_DATASET_FOLDER)
OBJECT_NAME = []

for c in range(len(OBJECT_CLASS)):
    OBJECT_NAME.append(os.listdir(LEARNING_DATASET_FOLDER + "/" + OBJECT_CLASS[c]))

TEACHING_DATA = "word/teaching_text.txt"
WORD_DICTIONARY = "word_dic.txt"
WORD_HIST = "histgram_word.txt"
IMG_HIST = "histgram_img.txt"
CODE_BOOK = "codebook.txt"