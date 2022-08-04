# 3D-Point-Clouds

3D点云SOTA方法,代码,论文,数据集(点云目标检测&amp;分割)

点云处理方法上主要包括两类方法

* 深度学习方法 [`python`]
* 传统上基于规则的方法 [`c++`]

@[双愚](https://github.com/HuangCongQing) , 若fork或star请注明来源

TODO

## 目录

#### 1 paper(code)

#### 2 Datasets

数据集基本处理: [数据集标注文件处理](https://github.com/HuangCongQing/Python#%E7%82%B9%E4%BA%91%E7%9B%B8%E5%85%B3%E5%A4%84%E7%90%86)

#### 3 点云可视化

点云可视化笔记和代码：https://github.com/HuangCongQing/Point-Clouds-Visualization

3D点云可视化的库有很多，你的选择可能是：

- pcl 点云可视化 [`c++`]
- ROS topic可视化  [`c++`]
- open3D [`python`]
- mayavi[`python`]
- matplolib [`python`]


#### 4 点云数据标注
数据标注工具总结：https://github.com/HuangCongQing/data-labeling-tools


## paper(code)

### 3D_Object_Detection
* One-stage
* Two-stage

#### One-stage

> Voxel-Net、SECOND、PointPillars、HVNet、DOPS、Point-GNN、SA-SSD、3D-VID、3DSSD

* Voxel-Net
* SECOND
* PointPillars
* HVNet
* DOPS
* Point-GNN
* SA-SSD
* 3D-VID
* 3DSSD

#### Two-stage


> F-pointNet、F-ConvNet、Point-RCNN、Part-A^2、PV-RCNN、Fast Point RCNN、TANet

* F-pointNet
* F-ConvNet
* Point-RCNN
* Part-A^2
* PV-RCNN
* Fast Point RCNN
* TANet

### 3D_Semantic_Segmentation
**PointNet** is proposed to learn per-point features using shared MLPs and global features using symmetrical pooling functions. Based on PointNet, a series of point-based networks have been proposed

>Point-based Methods: these methods can be roughly divided into pointwise MLP methods, point convolution methods, RNN-based methods, and graph-based methods

#### 1 pointwise MLP methods

> PointNet++，PointSIFT，PointWeb，ShellNet，RandLA-Net



PointNet++
PointSIFT
PointWeb
ShellNet
RandLA-Net




#### 2 point convolution methods

> PointCNN PCCN A-CNN ConvPoint pointconv KPConv DPC InterpCNN


* PointCNN
* PCCN
* A-CNN
* ConvPoint
* pointconv
* KPConv
* DPC
* InterpCNN


#### 3 RNN-based methods
> G+RCU  RSNet  3P-RNN  DAR-Net


* G+RCU  
* RSNet  
* 3P-RNN  
* DAR-Net


#### 4 graph-based methods

> DGCNN SPG SSP+SPG PyramNet GACNet PAG HDGCN  HPEIN SPH3D-GCN DPAM


* DGCNN
* SPG
* SSP+SPG
* PyramNet
* GACNet
* PAG
* HDGCN
* HPEIN
* SPH3D-GCN
* DPAM

### 3D_Instance Segmentation


## Datasets

### 数据集下载

* shell脚本下载方式: https://github.com/HuangCongQing/download_3D_dataset

- [https://hyper.ai/datasets](https://hyper.ai/datasets)
- [https://www.graviti.cn/open-datasets](https://www.graviti.cn/open-datasets)

> Graviti 收录了 400 多个高质量 CV 类数据集，覆盖无人驾驶、智慧零售、机器人等多种 AI 应用领域。举两个例子：
> 文章> [https://bbs.cvmart.net/topics/3346](https://bbs.cvmart.net/topics/3346)

- Google数据集搜索：[https://toolbox.google.com/datasetsearch](https://toolbox.google.com/datasetsearch)
- Datahub，分享高质量数据集平台：[https://datahub.io/](https://datahub.io/)
- 用于上传和查找数据集的机器学习数据集存储库：[https://www.webdoctx.com/www.mldata.org](https://www.webdoctx.com/www.mldata.org)
- datafountain收集数据集：[https://www.datafountain.cn/dataSets](https://www.datafountain.cn/dataSets)
- tinymind收集数据集：[https://www.tinymind.cn/sites#group_22](https://www.tinymind.cn/sites#group_22) 看到的一篇文章,里面有介绍很多数据集的：[世界上最有价值的不是石油而是数据(附数据资源下载链接)](https://mp.weixin.qq.com/s/Ao8SO9j2IPurl45Noy1dVw)
- [https://www.graviti.cn/open-datasets](https://www.graviti.cn/open-datasets)

## Datasets数据集汇总

[https://github.com/Yochengliu/awesome-point-cloud-analysis#---datasets](https://github.com/Yochengliu/awesome-point-cloud-analysis#---datasets)

- **[**[KITTI](http://www.cvlibs.net/datasets/kitti/)] The KITTI Vision Benchmark Suite. [`det.`]**常用
- [[ModelNet](http://modelnet.cs.princeton.edu/)] The Princeton ModelNet . [**`cls.`**]
- [[ShapeNet](https://www.shapenet.org/)] A collaborative dataset between researchers at Princeton, Stanford and TTIC. [**`seg.`**]
- [[PartNet](https://shapenet.org/download/parts)] The PartNet dataset provides fine grained part annotation of objects in ShapeNetCore. [**`seg.`**]
- [[PartNet](http://kevinkaixu.net/projects/partnet.html)] PartNet benchmark from Nanjing University and National University of Defense Technology. [**`seg.`**]
- **[**[**S3DIS**](http://buildingparser.stanford.edu/dataset.html#Download)**] The Stanford Large-Scale 3D Indoor Spaces Dataset. [`seg.`]**常用
- [[ScanNet](http://www.scan-net.org/)] Richly-annotated 3D Reconstructions of Indoor Scenes. [**`cls.`** **`seg.`**]
- [[Stanford 3D](https://graphics.stanford.edu/data/3Dscanrep/)] The Stanford 3D Scanning Repository. [**`reg.`**]
- [[UWA Dataset](http://staffhome.ecm.uwa.edu.au/~00053650/databases.html)] . [**`cls.`** **`seg.`** **`reg.`**]
- [[Princeton Shape Benchmark](http://shape.cs.princeton.edu/benchmark/)] The Princeton Shape Benchmark.
- [[SYDNEY URBAN OBJECTS DATASET](http://www.acfr.usyd.edu.au/papers/SydneyUrbanObjectsDataset.shtml)] This dataset contains a variety of common urban road objects scanned with a Velodyne HDL-64E LIDAR, collected in the CBD of Sydney, Australia. There are 631 individual scans of objects across classes of vehicles, pedestrians, signs and trees. [**`cls.`** **`match.`**]
- [[ASL Datasets Repository(ETH)](https://projects.asl.ethz.ch/datasets/doku.php?id=home)] This site is dedicated to provide datasets for the Robotics community with the aim to facilitate result evaluations and comparisons. [**`cls.`** **`match.`** **`reg.`** **`det`**]
- [[Large-Scale Point Cloud Classification Benchmark(ETH)](http://www.semantic3d.net/)] This benchmark closes the gap and provides a large labelled 3D point cloud data set of natural scenes with over 4 billion points in total. [**`cls.`**]
- [[Robotic 3D Scan Repository](http://asrl.utias.utoronto.ca/datasets/3dmap/)] The Canadian Planetary Emulation Terrain 3D Mapping Dataset is a collection of three-dimensional laser scans gathered at two unique planetary analogue rover test facilities in Canada.
- [[Radish](http://radish.sourceforge.net/)] The Robotics Data Set Repository (Radish for short) provides a collection of standard robotics data sets.
- [[IQmulus & TerraMobilita Contest](http://data.ign.fr/benchmarks/UrbanAnalysis/#)] The database contains 3D MLS data from a dense urban environment in Paris (France), composed of 300 million points. The acquisition was made in January 2013. [**`cls.`** **`seg.`** **`det.`**]
- [[Oakland 3-D Point Cloud Dataset](http://www.cs.cmu.edu/~vmr/datasets/oakland_3d/cvpr09/doc/)] This repository contains labeled 3-D point cloud laser data collected from a moving platform in a urban environment.
- [[Robotic 3D Scan Repository](http://kos.informatik.uni-osnabrueck.de/3Dscans/)] This repository provides 3D point clouds from robotic experiments，log files of robot runs and standard 3D data sets for the robotics community.
- [[Ford Campus Vision and Lidar Data Set](http://robots.engin.umich.edu/SoftwareData/Ford)] The dataset is collected by an autonomous ground vehicle testbed, based upon a modified Ford F-250 pickup truck.
- [[The Stanford Track Collection](https://cs.stanford.edu/people/teichman/stc/)] This dataset contains about 14,000 labeled tracks of objects as observed in natural street scenes by a Velodyne HDL-64E S2 LIDAR.
- [[PASCAL3D+](http://cvgl.stanford.edu/projects/pascal3d.html)] Beyond PASCAL: A Benchmark for 3D Object Detection in the Wild. [**`pos.`** **`det.`**]
- [[3D MNIST](https://www.kaggle.com/daavoo/3d-mnist)] The aim of this dataset is to provide a simple way to get started with 3D computer vision problems such as 3D shape recognition. [**`cls.`**]
- [[WAD](http://wad.ai/2019/challenge.html)] [[ApolloScape](http://apolloscape.auto/tracking.html)] The datasets are provided by Baidu Inc. [**`tra.`** **`seg.`** **`det.`**]
- [[nuScenes](https://d3u7q4379vrm7e.cloudfront.net/object-detection)] The nuScenes dataset is a large-scale autonomous driving dataset.用过
- [[PreSIL](https://uwaterloo.ca/waterloo-intelligent-systems-engineering-lab/projects/precise-synthetic-image-and-lidar-presil-dataset-autonomous)] Depth information, semantic segmentation (images), point-wise segmentation (point clouds), ground point labels (point clouds), and detailed annotations for all vehicles and people. [[paper](https://arxiv.org/abs/1905.00160)] [**`det.`** **`aut.`**]
- [[3D Match](http://3dmatch.cs.princeton.edu/)] Keypoint Matching Benchmark, Geometric Registration Benchmark, RGB-D Reconstruction Datasets. [**`reg.`** **`rec.`** **`oth.`**]
- [[BLVD](https://github.com/VCCIV/BLVD)] (a) 3D detection, (b) 4D tracking, (c) 5D interactive event recognition and (d) 5D intention prediction. [[ICRA 2019 paper](https://arxiv.org/abs/1903.06405v1)] [**`det.`** **`tra.`** **`aut.`** **`oth.`**]
- [[PedX](https://arxiv.org/abs/1809.03605)] 3D Pose Estimation of Pedestrians, more than 5,000 pairs of high-resolution (12MP) stereo images and LiDAR data along with providing 2D and 3D labels of pedestrians. [[ICRA 2019 paper](https://arxiv.org/abs/1809.03605)] [**`pos.`** **`aut.`**]
- [[H3D](https://usa.honda-ri.com/H3D)] Full-surround 3D multi-object detection and tracking dataset. [[ICRA 2019 paper](https://arxiv.org/abs/1903.01568)] [**`det.`** **`tra.`** **`aut.`**]
- [[Argoverse BY ARGO AI]](https://www.argoverse.org/) Two public datasets (3D Tracking and Motion Forecasting) supported by highly detailed maps to test, experiment, and teach self-driving vehicles how to understand the world around them.[[CVPR 2019 paper](http://openaccess.thecvf.com/content_CVPR_2019/html/Chang_Argoverse_3D_Tracking_and_Forecasting_With_Rich_Maps_CVPR_2019_paper.html)][**`tra.`** **`aut.`**]
- [[Matterport3D](https://niessner.github.io/Matterport/)] RGB-D: 10,800 panoramic views from 194,400 RGB-D images. Annotations: surface reconstructions, camera poses, and 2D and 3D semantic segmentations. Keypoint matching, view overlap prediction, normal prediction from color, semantic segmentation, and scene classification. [[3DV 2017 paper](https://arxiv.org/abs/1709.06158)] [[code](https://github.com/niessner/Matterport)] [[blog](https://matterport.com/blog/2017/09/20/announcing-matterport3d-research-dataset/)]
- [[SynthCity](https://arxiv.org/abs/1907.04758)] SynthCity is a 367.9M point synthetic full colour Mobile Laser Scanning point cloud. Nine categories. [**`seg.`** **`aut.`**]
- [[Lyft Level 5](https://level5.lyft.com/dataset/?source=post_page)] Include high quality, human-labelled 3D bounding boxes of traffic agents, an underlying HD spatial semantic map. [**`det.`** **`seg.`** **`aut.`**]
- **[**[**SemanticKITTI**](http://semantic-kitti.org/)**] Sequential Semantic Segmentation, 28 classes, for autonomous driving. All sequences of KITTI odometry labeled. [**[**ICCV 2019 paper**](https://arxiv.org/abs/1904.01416)**] [`seg.` `oth.` `aut.`]**常用
- [[NPM3D](http://npm3d.fr/paris-lille-3d)] The Paris-Lille-3D has been produced by a Mobile Laser System (MLS) in two different cities in France (Paris and Lille). [**`seg.`**]
- [[The Waymo Open Dataset](https://waymo.com/open/)] The Waymo Open Dataset is comprised of high resolution sensor data collected by Waymo self-driving cars in a wide variety of conditions. [**`det.`**]
- [[A*3D: An Autonomous Driving Dataset in Challeging Environments](https://github.com/I2RDL2/ASTAR-3D)] A*3D: An Autonomous Driving Dataset in Challeging Environments. [**`det.`**]
- [[PointDA-10 Dataset](https://github.com/canqin001/PointDAN)] Domain Adaptation for point clouds.
- [[Oxford Robotcar](https://robotcar-dataset.robots.ox.ac.uk/)] The dataset captures many different combinations of weather, traffic and pedestrians. [**`cls.`** **`det.`** **`rec.`**]

### 常用分割数据集

- **[**[**S3DIS**](http://buildingparser.stanford.edu/dataset.html#Download)**] The Stanford Large-Scale 3D Indoor Spaces Dataset. [`seg.`] [`常用`]
- **[**[**SemanticKITTI**](http://semantic-kitti.org/)**] Sequential Semantic Segmentation, 28 classes, for autonomous driving. All sequences of KITTI odometry labeled. [**[**ICCV 2019 paper**](https://arxiv.org/abs/1904.01416)**] [`seg.` `oth.` `aut.`] [`常用`]
- **Semantic3d**

### 常用分类数据集

todo

### 常用目标检测数据集

- **[**[KITTI](http://www.cvlibs.net/datasets/kitti/)] The KITTI Vision Benchmark Suite. [`det.`]**常用
- [[nuScenes](https://d3u7q4379vrm7e.cloudfront.net/object-detection)] The nuScenes dataset is a large-scale autonomous driving dataset.用过
- [[The Waymo Open Dataset](https://waymo.com/open/)] The Waymo Open Dataset is comprised of high resolution sensor data collected by Waymo self-driving cars in a wide variety of conditions. [**`det.`**]

## References

* https://github.com/timzhang642/3D-Machine-Learning
* https://github.com/victorphd/autonomous-vahicles-learning-resource
* https://github.com/Yochengliu/awesome-point-cloud-analysis
* https://github.com/NUAAXQ/awesome-point-cloud-analysis-2021
* https://github.com/QingyongHu/SoTA-Point-Cloud
* https://arxiv.org/abs/1912.12033 : Deep Learning for 3D Point Clouds: A Survey

## License

Copyright (c) [双愚](https://github.com/HuangCongQing). All rights reserved.

Licensed under the [MIT](./LICENSE) License.

---


微信公众号：**【双愚】**（huang_chongqing） 聊科研技术,谈人生思考,欢迎关注~

![image](https://user-images.githubusercontent.com/20675770/169835565-08fc9a49-573e-478a-84fc-d9b7c5fa27ff.png)

**往期推荐：**
1. [本文不提供职业建议，却能助你一生](https://mp.weixin.qq.com/s/rBR62qoAEeT56gGYTA0law)
2. [聊聊我们大学生面试](https://mp.weixin.qq.com/s?__biz=MzI4OTY1MjA3Mg==&mid=2247484016&idx=1&sn=08bc46266e00572e46f3e5d9ffb7c612&chksm=ec2aae77db5d276150cde1cb1dc6a53e03eba024adfbd1b22a048a7320c2b6872fb9dfef32aa&scene=178&cur_album_id=2253272068899471368#rd)
3. [清华大学刘知远：好的研究方法从哪来](https://mp.weixin.qq.com/s?__biz=MzI4OTY1MjA3Mg==&mid=2247486340&idx=1&sn=6c5f69bb37d91a343b1a1e7f6929ddae&chksm=ec2aa783db5d2e95ba4c472471267721cafafbe10c298a6d5fae9fed295f455a72f783872249&scene=178&cur_album_id=1855544495514140673#rd)
