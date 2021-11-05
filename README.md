# 'mlda' Package

The `mlda` enables the object categorization using MLDA (Multimodal Latent Dirichlet Allocation).   
Also, this package can work with ROS systems.  
Original MLDA code is here： [https://github.com/naka-tomo/MLDA-PY](https://github.com/naka-tomo/MLDA-PY)

![result_mlda](https://user-images.githubusercontent.com/74911522/140467824-07dfc742-df3b-48e4-adea-1d8f59b4fc48.png)

*   Maintainer: Shoichi Hasegawa ([hasegawa.shoichi@em.ci.ritsumei.ac.jp](mailto:hasegawa.shoichi@em.ci.ritsumei.ac.jp)).
*   Author: Shoichi Hasegawa ([hasegawa.shoichi@em.ci.ritsumei.ac.jp](mailto:hasegawa.shoichi@em.ci.ritsumei.ac.jp)).

**Content:**
* [Requirements](#requirements)
* [Folder](#folder)
* [Execution](#execution)
* [References](#references)

## Requirement
- Ubuntu：20.04LTS
- ROS：Noetic
- Python：3.8.10

```
roslib：1.15.7
rospy：1.15.11
cv-bridge：1.15.0
pathlib：1.0.1
future：0.18.2
futures：3.2.0
numpy：1.19.5
```

```
apt-get install python-tk
```

## Folder
- `mlda`： Multimodal LDA folder, which has Feature extraction code, execution MLDA code.
- `mlda_dataset_original`：Dataset of objects and images to learn.
- `README.md`： This file.


## Execution
1. Initialize data folder: `bash reset-mlda-output.bash`  
2. Create BoW:`python extract_word_bow.py`  
3. Create BoF:`python extract_img_bof.py`  
4. Execute MLDA:`python execute_node.py`  

If you want to change the dataset that trains the object, improve [mlda_dataset_original](https://github.com/Shoichi-Hasegawa0628/mlda_dataset_original/tree/rsj_experiment2) in data folder.

## References
*   Implementions of LDAs: [https://github.com/is0383kk/LDAs](https://github.com/is0383kk/LDAs)
*   Multimodal Object Dataset 165: [https://hp.naka-lab.org/subpages/mod165.html](https://hp.naka-lab.org/subpages/mod165.html)
*   MLDA: [https://github.com/naka-tomo/MLDA-PY](https://github.com/naka-tomo/MLDA-PY)










