# RoadCrackSegmentation

墙面裂缝检测

Unet that introduces the Global Context Block in Unet.  
  

## Usage  


### Train  
  `python train.py --train_images dataset/CRACK500/traincrop/ --train_annotations dataset/CRACK500/traincrop/ --epoch 100 --batch_size 32`  

### Test  
  `python test.py --save_weights_path 'checkpoint/'+ 'Unet/' + 'weights-099-0.1416-0.9787.h5 --vis False`  
  
## Results 
 

![](./imgs/20180822_081839_641_721.jpg)  


