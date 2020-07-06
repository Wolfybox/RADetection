# 【监控视频交通事故检测】  
##  [Road Accidents Detection in Surveillance Videos]  
# 【作者】  
## 杨泽华 Z.H.Yang   
# 【项目简介】  
## 【功能】  
检测监控视频中的交通事故发生所在位置。  
## 【性能】  
准确率：79.4%  
检测速度：43 FPS  
## 【评估指标】  
帧级AUC  
## 【数据集】  
### 名称  
URAD（UCF Road Accidents Dataset）  
### 来源  
UCF Crimes + CADP  
# 【运行环境与依赖】  
|类别|名称|版本|  
|-----|-----|-----|  
|os|Ubuntu or Windows|16.04(Ubuntu); Win10(Windows)|  
|编程语言|python|3.7|  
|深度学习框架|pytorch|1.5.1|  
|深度学习框架|torchvisioin|0.6.1|  
|计算机视觉框架|opencv-contrib-python|4.2.0.34|  
|图像处理库|pillow|7.1.2|  
|数据处理|numpy|1.18.5|  
|过程可视化|tqdm|4.46.1|  
# 输入与输出  
代码的输入与输出。如下所示：   
|名称|说明|  
|-----|-----|  
|输入|包含交通事故的监控视频；像素尺寸无限制|  
|输出|分数文件(Scores文件夹下的.npy文件)：每个视频输出一个分数文件，其内容为长度32的numpy array，分别代表视频32个段的预测异常分数|  
# 运行方式  
在terminal下运行以下命令：  
```shell  
$ cd project_dir  
$ python  detect.py [-h] [--gpu GPU] [--vdir VDIR] [--outdir OUTDIR] [--rgb_mdir RGB_MDIR] [--flow_mdir FLOW_MDIR]  
```  
参数说明：  
gpu：使用的GPU ID  
vdir：输入视频所在的文件夹  
outdir：分数输出的目标文件夹  
rgb_mdir：I3D RGB特征预训练模型文件  
flow_mdir：I3D FLOW特征预训练模型文件  
