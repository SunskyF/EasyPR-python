# ULPR (Universal License Plate Recognition)
ULPR(Universal License Plate Recognition)的设想是一个通用场景下的车牌识别系统。因为是从EasyPR出发，所以还是保留EasyPR-python的原库名。  
1. 用python写了一下EasyPR，但其中应该还是有bug，速度慢是detect部分很慢，有python本身的锅，也有我没有优化的锅  
2. 用deep的方法做检测和识别  

## Requirements
python 3  
tensorflow 1.5.0  
keras  
只在windows下进行了测试  

## Data
感谢[EasyPR](https://github.com/liuruoze/EasyPR)  
demo测试时使用了EasyPR的数据库  


## Download
训练easypr方法时，请下载easypr_train_data.zip放到data目录下  
测试时请下载data.zip放到data目录下
easypr的训练数据和各个模型的训练模型请从[百度云](https://pan.baidu.com/s/1bqmXEDD)上下载  
将模型文件：

1. whether\_car\_20180210T1049.zip
2. chars\_20180210T1038.zip
3. mrcnn\_20180212T2143.zip

解压放在output下。  

最后data文件夹下目录结构是  
├─demo  
├─easypr_train_data  
│  ├─chars  
│  └─whether_car  
├─general_test  
├─GDSL.txt  
└─使用说明.txt  
output文件夹下目录结构是  
├─chars_20180210T1038   
├─mrcnn\_20180212T2143  
└─whether_car_20180210T1049  

## TODO
[] multi-label的车牌识别  
[] 写博客  
[] 更好的根据mask获得车牌精确4个点的算法  
[] 整合训练时代码    
[] 轻量化  
[] end2end

## Done
[x] 重构  
[x] mask-rcnn  

## Train
可以参考scripts下的训练脚本  

## Demo  
切换不同方法时使用不同cfg即可，如将easypr.yml替换为maskrcnn.yml  
demo
```bash
# 用easypr的方法
python demo.py --cfg cfgs/test/easypr.yml --path data/demo/test.jpg
```

功能测试
```bash
python func_test.py --cfg cfgs/test/easypr.yml
```

批量测试（data目录下需要有general_test目录）
```bash
python accuracy_test.py --cfg cfgs/test/easypr.yml
```

## Tips
1. 训练mrcnn因为使用自己的数据，请注意一下数据格式    

## Reference
[EasyPR](https://github.com/liuruoze/EasyPR)  
[MaskRCNN](https://github.com/matterport/Mask_RCNN)
