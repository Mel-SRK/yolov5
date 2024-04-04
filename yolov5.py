#sys库，此库用来提供系统后息
import sys
#opencv库
import cv2
#torch.深度学习框架
import torch
#05库，此库用来提供操作系统功能
import os
#numpy.科学计算库
import numpy as np
#模型基础库中的模型加藏类
from models.common import DetectMultiBackend
#模型常用库中的输入图像检测及调整西数，N5西数，图像比例还原函数
from utils.general import (check_img_size,non_max_suppression,scale_boxes)
#torch中的设备加藏函数
from utils.torch_utils import select_device
# 绘制库中的注释器类颜色设置类
from utils.plots import Annotator,colors

#添加当前路径的上一级目录至Pyt♪hon搜索目录模块的路径集中
sys.path.append('..')
#由于比赛用车默认后动一个调用摄像头的服务，故在执行目标检测任务时要停掉此服务，此行为禁止默认服务的代码没有这行代码调用摄像头会冲突
os.system('sudo supervisorctl stop uniwiseRosCarTask')
##3、拍摄照片###
#通过opencv:获取比赛用车摄像头，序号为1O,此比赛用车固化了此唯一摄像头序号
cap = cv2.VideoCapture(10)
#创建两个量分别赋值为安颜读取的视频的两个返回值。第一个返回值为布尔值，第二返回值为三维矩阵数据形式的图像。
ret,frame=cap.read()
#frame = 'D:\资料\code\yolov5\yolov5\new.jpg'

#训练权重存放的路强
weights_path ='./yolov5n.pt'
#设置推理用设备，空为设置默认
device =''
#推理时进行多尺度，翻转等操作TTA)推理
augment = False
#框线条厚度
line_thickness=3
#置信度阈值
conf_thres=0.5
#恸NMS的iou威值
iou_thres=0.65
#设置只保留某一部分类别，例如阳或者023
classes=None
#进mms是否也去除不同类别之间的框，默认为False
agnostic_nms=True
#5、模型加载#
#通过select device
#定义执行脚本所用的设备
device = select_device(device)
#加扩Loat32模型
model = DetectMultiBackend(weights_path,device=device)
#创建变量存储模型的类别名称
names = model.names

#先BGR转换RGB,然后H,M,C转换为C,H,W
img = frame[:,:,::-1].transpose(2,0,1)
#ascontiguousarray)函数将一个内存不连续存储的数组转换为内存连续存储的数组
img = np.ascontiguousarray(img)
#np的数组转换为torch.的张量(array-→tensor)
img = torch.from_numpy(img).to(model.device)
#图徐设置为FLoat16
img = img.half()if model.fp16 else img.float()
#像素值从0-255转换为0.0-1.0
img/=255.0
#因为没有patch size,所以需要在最前面添加一个轴
if img.ndimension()==3:
    #例如3,640,480)÷(1,3,640,480)
    img = img.unsqueeze(0)

pred = model(img,augment=augment)[0]
pred = non_max_suppression(pred,conf_thres,iou_thres,classes=classes,agnostic=agnostic_nms)

#i代表第i个框，det是每个框里的信息
for i,det in enumerate(pred):
    #通过Annotator O创建类，类中存储了原图，框线厚度，类名
    annotator = Annotator(frame,line_width=line_thickness,example=str(names))
    #调整预测框的坐标：基于resize.+pad的图片的坐标基于原size图片的坐标
    det[:, :4] = scale_boxes(img.shape[2:],det[:4],frame.shape).round()
    #对det颠倒顺序后进行逼历
    for *xyxy,conf,cls in reversed(det):
        #创建变量c存储类别序号，用于在box label)方法中为不同类别随机不同颜色的框线
        c = int(cls)
        #创建变量Label存储类別及置信度
        label ='%s %.2f'(names[int(cls)],conf)
        #通过.box Label方法根据预测框位置、标签，及框线的颜色进行注释。框线颜色通过color)创建类根据类别设置
        annotator.box_label(xyxy,label,color=colors(c,True))
#创建变量maskImg存储注释后的图像
maskImg =annotator.result()
##10、保存图片##
#把最终的推理识别处理图像保存到指定位置：/home/儿amb/yolov!5-master,/images/youImg.jpg
os.system('mkdir ./images/')
cv2.imwrite('./images/youImg.jpg',maskImg)