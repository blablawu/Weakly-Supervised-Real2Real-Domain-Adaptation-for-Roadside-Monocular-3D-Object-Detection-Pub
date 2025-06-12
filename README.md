# Weakly-Supervised Real2Real Domain Adap- tation for Roadside Monocular 3D Object De- tection

## Introduction
This thesis focuses on the problems of real-to-real do- main adaptation in 3D object detection caused by variations in pitch angle and focal length of roadside cameras. MonoUNI is adopted as the baseline model, as it effectively mitigates these issues through its proposed normalized depth principle. Meanwhile, we employ WARM3D as our framework. To further assist the domain adaptation process, we introduce temporal infor- mation and corresponding constraints on top of the existing spatial constraints. Meanwhile, to improve the model’s ability of detecting out-of-distribution objects, we incorporate an off- the-shelf learnable 2D detector and 2D tracker. To test the performance of our method, we use three different KITTI-format datasets: Tumtraf-I, Rope3D and RCooper. Using 2D ground truth as weak labels, our method outperforms the Source-Only Baseline with improvement of 28.26%, 26.97% and 26.91% in AP3D@0.3 for easy, moderate and hard objects, respectively.

## News
- [x] ***[16.08.2024]*** create repo
- [x] ***[01.06.2025]*** We release code, log files
- [ ] The specific code cannot be disclosed at the moment. Thank you for your understanding.

## Installation
a. Clone this repository.
~~~
git clone https://github.com/blablawu/Weakly-Supervised-Real2Real-Domain-Adaptation-for-Roadside-Monocular-3D-Object-Detection-Pub.git
~~~

b. Install the dependent libraries as follows:
* Create a new env with conda
~~~
conda env create -f real2real_DA.yml
~~~

* Activate the env
~~~
conda activate real2real_DA
~~~

## Implementation for Off-the-shelf 2D Detector: YOLOV++ and 2D Tracker
### Installation of YOLOV++
The installation of YOLOV++ can be seen [**Here**](https://github.com/YuHengsss/YOLOV/tree/master). 

### Dataset-Preprocess: TUMTraf-I
  The initial directory will be as follows:  
  ```shell
  ├── TUMTraf-I
  │   ├── ImageSets
  │   ├── south1
  │   ├── south2
  │   ├── testing
  |      |── image_2
  |      |── label_2     
  |      |── calib
  |      |── extrinsics 
  │   ├── training
  |      |── image_2
  |      |── label_2      
  |      |── calib
  |      |── extrinsics 
  ```

* Step 1: Transfer kitti-format dataset into VOC-format
~~~
python yolov++_kitti2pascal_voc_Tumtraf.py
~~~  
After this step the directory will be as follows:
```shell
├── Tumtraf_VID_coarse
│   ├── Annotations ('.xml' format label files)
|      |── VID
|         |── train
|         |── val
│   ├── Data ('.jpg' format image files)  
|      |── VID
|         |── snippets ('.mp4' format video files)
|         |── train
|         |── val
│   ├── ImageSets
|      |── VID
|         |── train.txt
|         |── val.txt
```

* Step 2: Save the temporal image-sequence of train- and val-part into '.npy' files
~~~
python yolov++_Npy_generator.py
~~~

* Step 3. Convert VOC-format dataset into common_COCO-format (used for the training of yolov++'s base detector)
~~~
python yolov++_voc_to_common_coco.py
~~~

* Step 4. Convert VOC-format dataset into COCO_vid-format
~~~
python yolov++_voc_to_coco_vid.py
~~~

* Step 5. Convert COCO_vid-format dataset into ovis-format (used for the training of yolov++'s video detector)
~~~
python yolov++_coco_vid_to_ovis.py
~~~

### Dataset-Preprocess: RCooper
  The initial directory will be as follows:  
  ```shell
  ├── RCooper
  │   ├── training
  |      |── image_2
  |      |── label_2
  |      |── label_2_4cls
  |      |── box3d_depth_dense
  |      |── calib
  |      |── denorm
  |      |── extrinsics

  │   ├── validation
  |      |── image_2
  |      |── label_2
  |      |── label_2_4cls
  |      |── calib
  |      |── denorm
  |      |── extrinsics
  ```

* Step 1: Transfer kitti-format dataset into VOC-format
~~~
python yolov++_kitti2pascal_voc_RCooper.py
~~~  
After this step the directory will be as follows:
```shell
├── RCooper_VID_coarse
│   ├── Annotations ('.xml' format label files)
|      |── VID
|         |── train
|         |── val
│   ├── Data ('.jpg' format image files)  
|      |── VID
|         |── snippets ('.mp4' format video files)
|         |── train
|         |── val
│   ├── ImageSets
|      |── VID
|         |── train.txt
|         |── val.txt
```

* Step 2: Save the temporal image-sequence of train- and val-part into '.npy' files
~~~
python yolov++_Npy_generator.py
~~~

* Step 3. Convert VOC-format dataset into common_COCO-format (used for the training of yolov++'s base detector)
~~~
python yolov++_voc_to_common_coco.py
~~~

* Step 4. Convert VOC-format dataset into COCO_vid-format
~~~
python yolov++_voc_to_coco_vid.py
~~~

* Step 5. Convert COCO_vid-format dataset into ovis-format (used for the training of yolov++'s video detector)
~~~
python yolov++_coco_vid_to_ovis.py
~~~

### Quick start for YOLOV++
First, we train the base detector: YOLOX by the following command:
```bash
python tools/train.py \
  -f /home/heng/workspace/YOLOV/exps/swin_base/swin_tiny_vid.py \
  -c /home/heng/workspace/YOLOV/checkpoints/YOLOX/swin_tiny_coco_pretrained.pth \
  -b 1 -d 0 --fp16
```

Second, We can perform inference with the base detector by running the following command:
```bash
python tools/demo.py \
  -f /home/heng/workspace/YOLOV/exps/swin_base/swin_tiny_vid.py \
  -c YOLOX_outputs/swin_tiny_vid/best_ckpt.pth \
  --path, /home/heng/r02_tum_traffic_intersection_image_dataset_train_val_test_kitti/Tumtraf_VID_coarse/Data/VID/snippets/train/south_1/time_period_01.mp4 \
  --conf 0.25 \
  --nms 0.45 \
  --tsize 640 \
  --save_result \
  --device [cpu/gpu]
```
⚠️According to which video you want to inference, you need to modify the --path above

Third, we train the video detector: YOLOV++ by the following command:
```bash
python tools/vid_train.py \
  -f exps/customed_example/v++_SwinTiny_example.py \
  -c YOLOX_outputs/swin_tiny_vid/best_ckpt.pth \
  --fp16
```

Fourth, We can perform inference with the video detector by running the following command:
```bash
python tools/vid_demo.py \
  -f /home/heng/workspace/YOLOV/exps/customed_example/v++_SwinTiny_example.py \
  -c /home/heng/workspace/YOLOV/V++_outputs/v++_SwinTiny_example_Tumtraf/best_ckpt.pth \
  --path, /home/heng/r02_tum_traffic_intersection_image_dataset_train_val_test_kitti/Tumtraf_VID_coarse/Data/VID/snippets/train/south_1/time_period_01.mp4 \
  --conf 0.25 \
  --nms 0.5 \
  --tsize 576 \
  --save_result True
```
⚠️According to which video you want to inference, you need to modify the --path above

## Implementation for Off-the-shelf 2D Tracker: MCMOT (extension of FairMOT)
### Installation of MCMOT
The installation of FairMOT can be seen [**Here**](https://github.com/ifzhang/FairMOT). 

### Dataset-Preprocess: TUMTraf-I
  The initial directory will be as follows:

* Step 1: Generate the conveted format for Fairmot
~~~
python yolov++MCMOT_kitti2MOTvisdrone_Tumtraf.py
~~~  
```shell
├── Tumtraf_Track
│   ├── train
|      |── south_1_time_period_01
|         |── gt
|         |── img1
|         |── seqinfo.ini  
|      |── south_1_time_period_03    
|      |── south_1_time_period_04
|      |── south_2_time_period_01
|      |── south_2_time_period_03
|      |── south_2_time_period_04 
│   ├── val
|      |── south_1_time_period_02
|      |── south_2_time_period_02      
```

* Step 2: Generate the conveted format for MCMOT (used for the training of MCMOT)
~~~
python gen_dataset_Tumtraf.py
~~~
```shell
├── dataset/Tumtraf_fairmot
│   ├── ...
```

### Quick start for MCMOT
First, we train the 2D Tracker by the following command:
~~~
python /home/heng/workspace/MCMOT/src/train.py
~~~

Second, We can perform inference by running the following command:
```bash
python /home/heng/workspace/MCMOT/src/demo.py \
  --load_model /home/heng/workspace/MCMOT/exp/mot/default/model_120_tumtraf.pth \
  --input-video /home/heng/r02_tum_traffic_intersection_image_dataset_train_val_test_kitti/Tumtraf_VID_coarse/Data/VID/snippets/val/south_2_time_period_02.mp4
```
⚠️According to which video you want to inference, you need to modify the --input-video above

### RCooper
To be updated: the hyper-paramters of MCMOT still needs to be adjusted!


## Dataset-Preprocess
### TUMTraf-I (Target Domain)
- [x] Download the official TUMTraf-I dataset from [**Here**](https://innovation-mobility.com/tumtraf-dataset). 
- [x] We used initial version of TUMTraf-I dataset located at: 
    The directory will be as follows:  
    ```shell
    ├── TUMTraf-I
    │   ├── ImageSets
    │   ├── south1
    │   ├── south2
    │   ├── testing
    |      |── image_2
    |      |── label_2     
    |      |── calib
    |      |── extrinsics 
    │   ├── training
    |      |── image_2
    |      |── label_2      
    |      |── calib
    |      |── extrinsics 
    ```

* Step 1: Generate 3D Cube Depth for TUMtraf-I (need to adjust arg.split of this script into 'training' or 'testing',repectively)
~~~
python Tumtraf_Box3D_depth_dense_generator.py
~~~

* Step 2: Generate denorm-folder for TUMtraf-I (need to adjust arg.split of this script into 'training' or 'testing',repectively)
~~~
python Tumtraf_Denorm_generator.py
~~~
After this step the directory will be as follows:
```shell
├── Tumtraf_video
│   ├── train
|      |── image_2
|      |── label_2
|      |── calib
|      |── extrinsics
|      |── denorm
|      |── box3d_depth_dense

│   ├── val
|      |── image_2
|      |── label_2
|      |── calib
|      |── extrinsics
|      |── denorm
|      |── box3d_depth_dense
```

* Step 3: Matches images from two sources (KITTI format and temporal format) based on their MD5 hash values, and outputs the matched results into separate split files for training and testing datasets.
~~~
python Tumtraf_image_match_generator.py
~~~

* Step 4: Combine the temporal data together and regenerate the split file.
~~~
python Tumtraf_training_testing_combiner.py
~~~

* Step 5: Generate frame_id and tracking_id.
~~~
python Tumtraf_kitti_tracking_ID_generator.py
~~~

* Step 6: Fix the incorrect tracking_ID (Some Re-Identification issues were also manually modified during this part).
~~~
python Tumtraf_tracking_id_processor.py
~~~

* Step 7: Fill in the missing kitti label information of the object corresponding to a tracking_id in consecutive frames.
~~~
python Tumtraf_linear_interpolation_kitti_label.py
~~~

* Step 8: Replace the height, width, and length of the object with the median value for those objects whose height, width, and length change significantly(out of threshold)
~~~
python hwl_statistic.py
~~~

* Step 9: Redistribute the original training/testing dataset into 6 different video frames
~~~
python Tumtraf_unordered2video_sequence.py
~~~
After this step the directory will be as follows:
```shell
├── Tumtraf_video
│   ├── train
|      |── image_2
|         |── south_1
|         |── south_2
|      |── label_2_kitti_4cls
|      |── label_2_kitti
|      |── box3d_depth_dense
|      |── calib
|      |── denorm
|      |── extrinsics

│   ├── val
|      |── image_2
|         |── south_1_seq_0001
|         |── south_2_seq_0001
|      |── label_2_kitti_4cls
|      |── label_2_kitti
|      |── box3d_depth_dense
|      |── calib
|      |── denorm
|      |── extrinsics
```

* Step 10: Generate 'train.json' and 'val.json' for training with temporal information
~~~
python Tumtraf_Warm3d_imglist_generator.py
~~~

* Step 11: Generate 2D pseudo labels from 2D Detector: YOLOV++
```bash
python vid_demo.py \
  -f /home/heng/workspace/YOLOV/exps/customed_example/v++_SwinTiny_example.py \
  -c /home/heng/workspace/YOLOV/V++_outputs/v++_SwinTiny_example_Tumtraf/best_ckpt.pth \
  --path /home/heng/r02_tum_traffic_intersection_image_dataset_train_val_test_kitti/Tumtraf_VID_coarse/Data/VID/snippets/train/south_1/time_period_01.mp4 \
  --conf 0.25 --nms 0.5 --tsize 576 --save_result
```

* Step 12: Copy 2D pseudo labels from YOLOV++ to TUMTraf-I dataset
~~~
python Tumtraf_Copy_2D_pseudo_label_from_2Dpredictor.py
~~~

* Step 13: Generate 2D pseudo labels from 2D Tracker: MCMOT
```bash
python demo.py \
  --load_model /home/heng/workspace/MCMOT/exp/mot/default/model_120.pth \
  --input-video /home/heng/r02_tum_traffic_intersection_image_dataset_train_val_test_kitti/Tumtraf_VID_coarse/Data/VID/snippets/val/south_1_time_period_02.mp4
```

* Step 14: Copy 2D pseudo labels from MCMOT to TUMTraf-I dataset
~~~
python Tumtraf_Copy_2D_pseudo_label_from_2Dpredictor.py
~~~
The final directory will be as follows:
```shell
├── Tumtraf_video
│   ├── train
|      |── image_2
|      |── label_2_kitti_coarse
|      |── label_2_kitti_fine
|      |── pseudo_2D_label_coarse
|      |── pseudo_2D_label_withID_coarse
|      |── box3d_depth_dense
|      |── calib
|      |── denorm
|      |── extrinsics

│   ├── val
|      |── image_2
|      |── label_2_kitti_4cls
|      |── label_2_kitti
|      |── pseudo_2D_label_coarse
|      |── pseudo_2D_label_withID_coarse
|      |── box3d_depth_dense
|      |── calib
|      |── denorm
|      |── extrinsics
```

* ⚠️ Details can be seen in file README_dataset_process.md from the path: /home/heng/workspace/MonoUNI/Dataset_Tumtraf_I_process_scripts


### RCooper (Target Domain)
- [x] Please check the bottom of this page [website](https://www.t3caic.com/qingzhen/) to download the data. As shown in the figure bellow.

    <div style="text-align:center">
    <img src="imgs/dataset_page_instruction.jpg" width="700" alt="" class="img-responsive">
    </div>

    After downloading the data, please put the data in the following structure:
    ```shell
    ├── RCooper
    │   ├── calib
    |      |── lidar2cam
    |      |── lidar2world
    │   ├── data
    |      |── folders named specific scene index
    │   ├── labels
    |      |── folders named specific scene index
    │   ├── corridor
    │   ├── intersection
    ```

* Step 1: merge the training and validation sets of the corridor and intersection scenarios.
~~~
python RCooper_MonoUNI_data_transformer.py
~~~

* Step 2: Generate 3D Cube Depth for RCooper
~~~
python RCooper_Box3D_depth_dense_generator.py
~~~

* Step 3: Generate denorm-folder for RCooper (need to adjust arg.split and arg.which_scenario of this script,repectively)
~~~
python RCooper_Denorm_generator.py
~~~

* Step 4: Merge the files from different scenarios('corridor' and 'intersection') of dataset RCooper together into 'training' and 'validation'
~~~
python RCooper_scenarios_merge.py
~~~

* Step 5: Split the original training and validation sets into several different video frames
~~~
python RCooper_unordered2video_sequence.py
~~~
After this step the directory will be as follows:
```shell
├── RCooper_video
│   ├── train
|      |── image_2
|      |── label_2_kitti_coarse
|      |── label_2_kitti_fine
|      |── box3d_depth_dense
|      |── calib
|      |── denorm
|      |── extrinsics

│   ├── val
|      |── image_2
|      |── label_2_kitti_coarse
|      |── label_2_kitti_fine
|      |── box3d_depth_dense
|      |── calib
|      |── denorm
|      |── extrinsics
```

* Step 6: Generate 'train.json' and 'val.json' for training with temporal information
~~~
python RCooper_Warm3d_imglist_generator.py
~~~

* Step 7: Generate 2D pseudo labels from 2D Detector: YOLOV++
```bash
python vid_demo.py \
  -f /home/heng/workspace/YOLOV/exps/customed_example/v++_SwinTiny_example.py \
  -c /home/heng/workspace/YOLOV/V++_outputs/v++_SwinTiny_example/best_ckpt.pth \
  --path /home/heng/RCooper/RCooper_VID_coarse/Data/VID/snippets/train/corridor/seq_0000.mp4 \
  --conf 0.25 --nms 0.5 --tsize 576 --save_result
```

* Step 8: Copy 2D pseudo labels from YOLOV++ to TUMTraf-I dataset
~~~
python RCooper_Copy_2D_pseudo_label_from_2Dpredictor.py
~~~
After this step the directory will be as follows:
```shell
├── RCooper_video
│   ├── train
|      |── image_2
|      |── label_2_kitti_coarse
|      |── label_2_kitti_fine
|      |── pseudo_2D_label_coarse
|      |── box3d_depth_dense
|      |── calib
|      |── denorm
|      |── extrinsics

│   ├── val
|      |── image_2
|      |── label_2_kitti_coarse
|      |── label_2_kitti_fine
|      |── pseudo_2D_label_coarse
|      |── box3d_depth_dense
|      |── calib
|      |── denorm
|      |── extrinsics
```

* ⚠️ Details can be seen in file README_dataset_process.md from the path: /home/heng/workspace/MonoUNI/Dataset_RCooper_process_scripts


### Rope3D (Source Domain)
- [x] Download the official Rope3D dataset from [**Here**](https://pan.baidu.com/s/1Tt014qMNcDxAMCkEWH_EZQ?pwd=d1yd).  
    ~~~
    tar -zxvf Rope3D_data.tar.gz
    ~~~
    The directory will be as follows:  
    ```shell
    ├── Rope3D
    │   ├── box3d_depth_dense
    │   ├── calib
    │   ├── denorm
    │   ├── extrinsics
    │   ├── image_2
    │   ├── ImageSets
    │   ├── label_2
    │   ├── label_2_4cls_filter_with_roi_for_eval
    │   ├── label_2_4cls_for_train
    ```

## Train
    Modify the configuration parameters in config.yaml according to training requirements
    ⚠️ Details about training can be seen in config.yaml
    ~~~
    python train_val.py
    ~~~

## Eval
    Modify the configuration parameters in config.yaml according to evaluation requirements
    ⚠️ Details about evaluation can be seen in config.yaml
    ~~~
    python train_val.py --evaluate
    ~~~


## Detecting demo on TUMTraf-I dataset
![TUMTraf-I](https://github.com/blablawu/Weakly-Supervised-Real2Real-Domain-Adaptation-for-Roadside-Monocular-3D-Object-Detection/blob/main/gifs/TUMTraf-I_demo.gif)

## Detecting demo on RCooper dataset
![RCooper_corridor](https://github.com/blablawu/Weakly-Supervised-Real2Real-Domain-Adaptation-for-Roadside-Monocular-3D-Object-Detection/blob/main/gifs/RCooper_Corridor_demo.gif)


## Acknowledgements
Many thanks to following codes that help us a lot in building this codebase:
- [MonoUNI](https://github.com/Traffic-X/MonoUNI) 
- [WARM3D](https://github.com/WARM-3D/WARM-3D)
- [Rope3D](https://github.com/liyingying0113/rope3d-dataset-tools)
- [TUMTraf-I](https://github.com/tum-traffic-dataset/tum-traffic-dataset-dev-kit)
- [RCooper](https://github.com/AIR-THU/DAIR-RCooper/tree/main)
- [YOLOV++](https://github.com/YuHengsss/YOLOV/tree/master)
- [MCMOT](https://github.com/CaptainEven/MCMOT)

