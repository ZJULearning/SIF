# SIF: Self-Inspirited Feature Learning for Person Re-Identification
## Introduction

SIF is a new training method for person re-identification (Re-ID) networks. Given an existing Re-ID network, an auxiliary branch is added into the network only in the training stage, while the structure of the original network stays unchanged during the testing stage. 

This project is the implementation of our IEEE TIP paper - [SIF: Self-Inspirited Feature Learning for Person Re-Identification](https://ieeexplore.ieee.org/document/9024230) on some commonly used baseline networks. Our code is adapted from the open-reid library (https://github.com/Cysu/open-reid).

## Datasets
* [Market-1501](http://www.liangzheng.com.cn/Project/project_reid.html)
  
    Download using: 
        
      wget http://188.138.127.15:81/Datasets/Market-1501-v15.09.15.zip <path/to/where/you/want>
      unzip <path/to/>/Market-1501-v15.09.15.zip
  
* [CUHK03](http://www.ee.cuhk.edu.hk/~xgwang/CUHK_identification.html)

  1. Download cuhk03 dataset from [here](http://www.ee.cuhk.edu.hk/~xgwang/CUHK_identification.html)
  2. Unzip the file and you will get the cuhk03_release dir which include cuhk-03.mat
  3. Download "cuhk03_new_protocol_config_detected.mat" from [here](https://github.com/zhunzhong07/person-re-ranking/tree/master/evaluation/data/CUHK03) and put it with cuhk-03.mat. We need this new protocol to split the dataset.
  ```
  python utils/transform_cuhk03.py --src <path/to/cuhk03_release> --dst <path/to/save>
  ```
  NOTICE: You need to change num_classes in network depend on how many people in your train dataset! e.g. 751 in Market1501.

The data structure should look like:
    
  ```
  data/
      bounding_box_train/
      bounding_box_test/
      query/
      train.txt   
      val.txt
      query.txt
      gallery.txt
  ```
  Here each *.txt file consists lines of the format: image file name, person id, camera id.
  train.txt consists images from bounding_box_train/, val.txt and query.txt consists images from query/, and gallery.txt consists images from bounding_box_test/.

## RUN
### Prerequisites

+ cudnn 7
+ CUDA 9
+ Pytorch v0.4.1
+ Python 2.7
+ torchvision
+ scipy
+ numpy
+ scikit_learn

### Baseline ReID methods

+ [ResNet](https://arxiv.org/abs/1512.03385). We choose two configurations: ResNet50 and ResNet152.
+ [DenseNet](https://arxiv.org/abs/1608.06993). We choose two configurations: DenseNet121 and DenseNet161.

### Train
We provie two training methods: plain and sif.

* Train using plain method (traditional SGD optimization)
  ```
  CUDA_VISIBLE_DEVICES=0 python train_eval.py --arch mgn --mode train --usegpu --project_name 'temp_project' --data_path <path/to/Market-1501-v15.09.15> --lr 2e-4 --batchid 4 --epoch 450
  ```
* Train using sif method (our proposed SIF optimization)
  ```
  CUDA_VISIBLE_DEVICES=0 python train_eval.py --arch mgn --mode train --usegpu --project_name 'temp_project' --data_path <path/to/Market-1501-v15.09.15> --lr 2e-4 --batchid 4 --epoch 450
  ``` 
### Results

<table>
  <tr>
    <th>Models + Training_method</th> 
    <th colspan="2">Market-1501</th>
    <th colspan="2">DukeMTMC-reID</th>
    <th colspan="2">CUHK03(Detected)</th>
    <th colspan="2">CUHK03(Labelled)</th>
  </tr>
  <tr>
    <td></td>
    <td>mAP</td>
    <td>CMC-1</td>
    <td>mAP</td>
    <td>CMC-1</td>
    <td>mAP</td>
    <td>CMC-1</td>
    <td>mAP</td>
    <td>CMC-1</td>
  </tr>
  <tr>
    <td>ResNet50 + SGD</td>
    <td>70.51</td>
    <td>89.34</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
  </tr>
  <tr>
    <td>ResNet50 + SIF</td>
    <td>75.70</td>
    <td>91.20</td>
    <td>63.80</td>
    <td>80.50</td>
    <td>38.60</td>
    <td>41.70</td>
    <td>41.00</td>
    <td>44.40</td>
  </tr>
  <tr>
    <td>DenseNet121 + SGD</td>
    <td>77.30</td>
    <td>92.40</td>
    <td>65.30</td>
    <td>81.90</td>
    <td>54.20</td>
    <td>61.30</td>
    <td>-</td>
    <td>-</td>
  </tr>
  <tr>
    <td>DenseNet121 + SIF</td>
    <td>81.60</td>
    <td>93.80</td>
    <td>69.20</td>
    <td>83.30</td>
    <td>57.50</td>
    <td>63.70</td>
    <td>-</td>
    <td>-</td>
  </tr>
  <tr>
    <td>ResNet152 + SGD</td>
    <td>86.90</td>
    <td>95.70</td>
    <td>78.40</td>
    <td>88.70</td>
    <td>66.00</td>
    <td>66.80</td>
    <td>67.40</td>
    <td>68.00</td>
  </tr>
  <tr>
    <td>ResNet152 + SIF</td>
    <td>85.80</td>
    <td>94.60</td>
    <td>77.07</td>
    <td>87.70</td>
    <td>69.41</td>
    <td>71.64</td>
    <td>72.96</td>
    <td>74.07</td>
  </tr>
  <tr>
    <td>DenseNet161 + SGD</td>
    <td>87.34</td>
    <td>94.83</td>
    <td>79.16</td>
    <td>89.36</td>
    <td>74.22</td>
    <td>76.14</td>
    <td>77.31</td>
    <td>79.79</td>
  </tr>
  <tr>
    <td>DenseNet161 + SIF</td>
    <td>87.34</td>
    <td>94.83</td>
    <td>79.16</td>
    <td>89.36</td>
    <td>74.22</td>
    <td>76.14</td>
    <td>77.31</td>
    <td>79.79</td>
  </tr>
</table>

## Reference

Reference to cite when you use SIF in a research paper:

Wei, Long, Zhenyong Wei, Zhongming Jin, Zhengxu Yu, Jianqiang Huang, Deng Cai, Xiaofei He, and Xian-Sheng Hua. "SIF: Self-Inspirited Feature Learning for Person Re-Identification." IEEE Transactions on Image Processing 29 (2020): 4942-4951.

## License
PTL is MIT-licensed.
