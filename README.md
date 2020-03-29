# SIF: Self-Inspirited Feature Learning for Person Re-Identification
## Introduction

SIF is a new training method for person re-identification (Re-ID) networks. Given an existing Re-ID network, an auxiliary branch is added into the network only in the training stage, while the structure of the original network stays unchanged during the testing stage. 

This project is the implementation of our IEEE TIP paper - [SIF: Self-Inspirited Feature Learning for Person Re-Identification](https://ieeexplore.ieee.org/document/9024230) on some commonly used baseline networks. Our code is adapted from the open-reid library (https://github.com/Cysu/open-reid).


## Performance

### Datasets
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
  
        
### Baseline ReID methods

+ [ResNet](https://arxiv.org/abs/1512.03385). We choose two configurations: ResNet50 and ResNet152.
+ [DenseNet](https://arxiv.org/abs/1608.06993). We choose two configurations: DenseNet121 and DenseNet161.
### Results

<table>
  <tr>
    <th>Model</th> 
    <th>Method</th>
    <th colspan="2">Market-1501</th>
    <th colspan="2">DukeMTMC-reID</th>
    <th colspan="2">CUHK03(Detected)</th>
    <th colspan="2">CUHK03(Labelled)</th>
  </tr>
  <tr>
    <td></td>
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
    <td>DML</td>
    <td>DML</td>
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
    <td>HA-CNN</td>
    <td>HA-CNN</td>
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
    <td>PCB</td>
    <td>PCB</td>
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
    <td>PCB+RPP</td>
    <td>PCB+RPP</td>
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
    <td>MGN</td>
    <td>MGN</td>
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
    <td>MGN(reproduced)</td>
    <td>MGN(reproduced)</td>
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
    <td><b>MGN_PTL</b></td>
    <td><b>MGN_PTL</b></td>
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
    <td><b>MGN_PTL</b></td>
    <td><b>MGN_PTL</b></td>
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



NOTICE: The MGN(reproduced) is the reproduction of [MGN](https://arxiv.org/pdf/1804.01438.pdf). To our best knowledge, the official implementation of MGN has not released yet. Hence, the **MGN_PTL**
network used the MGN(reproduced) as backbone network. The code for MGN(reproduced) is in **mgn.py** 

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

### GPU usage

We used one Tesla P100 GPU in our experiments
* To run the MGN with batchid=4 and batchimage=4 cost 7819 MiB
* To run the MGN_PTL with batchid=4 and batchimage=4 cost 8819 MiB

### Train
You can specify more parameters in opt.py

* Train MGN_PTL
  ```
  CUDA_VISIBLE_DEVICES=0 python train_eval.py --arch mgn_ptl --mode train --usegpu --project_name 'temp_project' --data_path <path/to/Market-1501-v15.09.15> --lr 2e-4 --batchid 4 --epoch 450
  ```
* Train MGN
  ```
  CUDA_VISIBLE_DEVICES=0 python train_eval.py --arch mgn --mode train --usegpu --project_name 'temp_project' --data_path <path/to/Market-1501-v15.09.15> --lr 2e-4 --batchid 4 --epoch 450
  ```

### Evaluate
Use pretrained weight or your trained weight

* Evaluate MGN_PTL
  ```
  CUDA_VISIBLE_DEVICES=0 python train_eval.py --arch mgn_ptl --mode evaluate --usegpu --weight <path/to/weight/weight_name.pt> --data_path <path/to/Market-1501-v15.09.15>
   ```
* Evaluate MGN
  ```
  CUDA_VISIBLE_DEVICES=0 python train_eval.py --arch mgn --mode evaluate --usegpu --weight <path/to/weight/weight_name.pt> --data_path <path/to/Market-1501-v15.09.15>
   ```
   
## Reference

Reference to cite when you use SIF in a research paper:
	@article{wei2020sif,
  		title={SIF: Self-Inspirited Feature Learning for Person Re-Identification},
  		author={Wei, Long and Wei, Zhenyong and Jin, Zhongming and Yu, Zhengxu and Huang, Jianqiang and Cai, Deng and He, Xiaofei and Hua, Xian-Sheng},
  		journal={IEEE Transactions on Image Processing},
  		volume={29},
  		pages={4942--4951},
  		year={2020},
  		publisher={IEEE}
	}

    @inproceedings{ijcai2019-586,
      title     = {Progressive Transfer Learning for Person Re-identification},
      author    = {Yu, Zhengxu and Jin, Zhongming and Wei, Long and Guo, Jishun and Huang, Jianqiang and Cai, Deng and He, Xiaofei and Hua, Xian-Sheng},
      booktitle = {Proceedings of the Twenty-Eighth International Joint Conference on Artificial Intelligence, {IJCAI-19}},
      publisher = {International Joint Conferences on Artificial Intelligence Organization},             
      pages     = {4220--4226},
      year      = {2019},
      month     = {7},
      doi       = {10.24963/ijcai.2019/586},
      url       = {https://doi.org/10.24963/ijcai.2019/586},
      }
## License
PTL is MIT-licensed.
