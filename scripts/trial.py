#! /usr/bin/env python
# -*- coding: utf-8 -*-

with open("/root/RULO/catkin_ws/src/spco2_mlda_problog/mlda/data/mlda_dataset_original/rsj_exp/bottle/coffee/word/teaching_text.txt") as f:
    current_word = [s.strip() for s in f.readlines()]
print(current_word)

current_word.append("apple")
with open("/root/RULO/catkin_ws/src/spco2_mlda_problog/mlda/data/mlda_dataset_original/rsj_exp/bottle/coffee/word/teaching_text.txt", mode='w') as f:
    f.write('\n'.join(current_word))
print(type(current_word))