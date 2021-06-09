# 'mlda' Package

The `mlda` enables the object categorization using MLDA.

*   Maintainer: Shoichi Hasegawa ([hasegawa.shoichi@em.ci.ritsumei.ac.jp](mailto:hasegawa.shoichi@em.ci.ritsumei.ac.jp)).
*   Author: Shoichi Hasegawa ([hasegawa.shoichi@em.ci.ritsumei.ac.jp](mailto:hasegawa.shoichi@em.ci.ritsumei.ac.jp)).

**Content:**
*   [Requirements](#requirements)
*   [Launch](#launch)
*   [References](#references)

## Requirement

Python2.X and the following modules are required

```python
import numpy as np
import random
import math
import pylab
import pickle
import os
import sys
import cv2
import glob
import codecs
```


## Launch
1. Write sentences in `/MLDA/bow/text.txt`  
2. Create BoW:`python /MLDA/bow/bow.py`  
3. Create BoF:`python /MLDA/bof/bof.py`  
4. `python /MLDA/mlda.py`  


- `/MLDA/mlda.py`:MLDA by using Collapsed Gibbs sampler with Python.You need to decide the number of `topic`.
```python
def main():
    topic = 3
    data = []
    data.append( np.loadtxt( "./bof/histogram_v.txt" , dtype=np.int32) )
    data.append( np.loadtxt( "./bow/histogram_w.txt" , dtype=np.int32)*5 )
    mlda( data, topic, 100, "learn_result" )

    data[1] = None
    mlda( data, topic, 10, "recog_result" , "learn_result" )
```
**※Before running this script, you need to run `/MLDA/bow/bow.py` and `/MLDA/bof/bof.py` which create BoW and BoF file**
- `/MLDA/bow/bow.py`: It can generate **BoW file** which is used by `/MLDA/mlda.py` from `/MLDA/bow/text.txt`
- `/MLDA/bow/text.txt`: You can write  sentences.  
**※Sentences must be separated by spaces for each word.**
    - *For example*
    ```
    内藤 は 彼女 が できない
    内藤 は 理想 が 高すぎる
    あの 店 の ラーメン は おいしい
    おいしい 店 の ごはん を 食べる
    室 は ずっと 踊って いる
    室 は 顔芸 を して いる
    ```

- `/MLDA/bof/bof.py`: It can generate **BoF file** which is used by `/MLDA/mlda.py` from `/MLDA/bof/images/*.png`




## References
*   Implementions of LDAs: [https://github.com/is0383kk/LDAs](https://github.com/is0383kk/LDAs)
*   Multimodal Object Dataset 165: [https://hp.naka-lab.org/subpages/mod165.html](https://hp.naka-lab.org/subpages/mod165.html)
*   MLDA: [https://github.com/naka-tomo/MLDA-PY](https://github.com/naka-tomo/MLDA-PY)










