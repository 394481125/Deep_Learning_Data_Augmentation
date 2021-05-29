import os
import cv2
import math
import argparse
import numpy as np
import copy
from skimage import exposure
from skimage.util import random_noise
import random
import datetime
import xml.etree.ElementTree as ET


import imageio
import imgaug as ia
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
from imgaug import augmenters as iaa


"""
YOLO图像增广
"""


class ImageAugmentation:
    def __init__(self):
        pass

    def getBoxes(self, image_name):
        """
        根据XML标注文件得到标注列表[x_min, y_min, x_max, y_max, cat_name]的列表
        :param image_name:
        :return:
        """
        tree = ET.parse(image_name + '.xml')
        root = tree.getroot()
        boxes = []
        for object in root.findall('object'):
            temp_list = []
            name = object.find('name').text
            for coordinate in object.find('bndbox'):
                temp_list.append(int(coordinate.text))
            temp_list.append(name)
            boxes.append(temp_list)
        # print(boxes)
        return boxes

    def saveXML(self, image_name, xml_name, boxes, shape1, shape0):
        print("xml name ====================================" + xml_name)
        print("image name ====================================" + xml_name)
        folder = ET.Element('folder')
        folder.text = 'image'

        filename = ET.Element('filename')
        filename.text = image_name

        path = ET.Element('path')
        curr_path = os.getcwd()

        path.text = curr_path + '\\image\\' + image_name

        source = ET.Element('source')
        database = ET.SubElement(source, 'database')
        database.text = 'Unknown'

        size = ET.Element('size')
        width = ET.SubElement(size, 'width')

        width.text = str(shape1)
        height = ET.SubElement(size, 'height')
        height.text = str(shape0)
        depth = ET.SubElement(size, 'depth')
        depth.text = '3'
        segmented = ET.Element('segmented')
        segmented.text = '0'

        root = ET.Element('annotation')
        root.extend((folder, filename, path))
        root.extend((source, size, segmented))

        for box in boxes:
            object = ET.Element('object')
            name = ET.SubElement(object, 'name')
            name.text = box[4]
            pose = ET.SubElement(object, 'pose')
            pose.text = 'Unspecified'
            truncated = ET.SubElement(object, 'truncated')
            truncated.text = '0'
            difficult = ET.SubElement(object, 'difficult')
            difficult.text = '0'
            bndbox = ET.SubElement(object, 'bndbox')
            xmin = ET.SubElement(bndbox, 'xmin')
            xmin.text = str(box[0])
            ymin = ET.SubElement(bndbox, 'ymin')
            ymin.text = str(box[1])
            xmax = ET.SubElement(bndbox, 'xmax')
            xmax.text = str(box[2])
            ymax = ET.SubElement(bndbox, 'ymax')
            ymax.text = str(box[3])
            root.extend((object,))

        tree = ET.ElementTree(root)
        tree.write(xml_name)

        tree = ET.parse(xml_name)  # 解析movies.xml这个文件
        root = tree.getroot()  # 得到根元素，Element类
        self.pretty_xml(root, '\t', '\n')  # 执行美化方法
        tree.write(xml_name)

    def pretty_xml(self, element, indent, newline, level=0):  # elemnt为传进来的Elment类，参数indent用于缩进，newline用于换行
        if element:  # 判断element是否有子元素
            if (element.text is None) or element.text.isspace():  # 如果element的text没有内容
                element.text = newline + indent * (level + 1)
            else:
                element.text = newline + indent * (level + 1) + element.text.strip() + newline + indent * (level + 1)
                # else:  # 此处两行如果把注释去掉，Element的text也会另起一行
                # element.text = newline + indent * (level + 1) + element.text.strip() + newline + indent * level
        temp = list(element)  # 将element转成list
        for subelement in temp:
            if temp.index(subelement) < (len(temp) - 1):  # 如果不是list的最后一个元素，说明下一个行是同级别元素的起始，缩进应一致
                subelement.tail = newline + indent * (level + 1)
            else:  # 如果是list的最后一个元素， 说明下一行是母元素的结束，缩进应该少一个
                subelement.tail = newline + indent * level
            self.pretty_xml(subelement, indent, newline, level=level + 1)  # 对子元素进行递归操作

    def changeImages(self, folder, function_name, image_name, n):
        if function_name == "crop":
            function = self.__cropImage1
        elif function_name == "tran":
            function = self.__translationImage
        elif function_name == "light":
            function = self.__changeLightofImage
        elif function_name == "noise":
            function = self.__addNoiseToImage
        elif function_name == "rotate":
            function = self.__rotateImage
        elif function_name == "flip":
            function = self.__flipImage

        image = cv2.imread(image_name + '.jpg')
        boxes = self.getBoxes(image_name)
        for i in range(1, n + 1):
            print(function_name + " image #" + str(i))
            change_img, change_boxes = function(copy.deepcopy(image), copy.deepcopy(boxes))
            print("Old boxes: ", boxes)
            print("New boxes: ", change_boxes)
            current_time = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
            current_num = str(random.randint(0, 9999))
            save_image_name = folder + '/' + current_time + '_' + str(i) + current_num + '.jpg'
            save_xml_name = folder + '/' + current_time + '_' + str(i) + current_num + '.xml'
            print(image_name)
            print("save image name: " + save_image_name)
            print("save xml name:   " + save_xml_name)
            cv2.imwrite(save_image_name, change_img)
            self.saveXML(save_image_name, save_xml_name, change_boxes, change_img.shape[1], change_img.shape[0])
            print("Save new image to current path: " + save_image_name)
            print("Save new xml to current path:   " + save_xml_name)
            print("\n")

    # 1 裁切
    def __cropImage(self, img, boxes):
        """
        裁切
        :param img: 图像
        :param bboxes: 该图像包含的所有boundingboxes，一个list，每个元素为[x_min,y_min,x_max,y_max]
        :return: crop_img：裁剪后的图像；crop_bboxes：裁剪后的boundingbox的坐标，list
        """
        # 裁剪图像
        w = img.shape[1]
        h = img.shape[0]

        x_min = w
        x_max = 0
        y_min = h
        y_max = 0

        # 最小区域
        for bbox in boxes:
            x_min = min(x_min, bbox[0])
            y_min = min(y_min, bbox[1])
            x_max = max(x_max, bbox[2])
            y_max = max(y_max, bbox[3])
            name = bbox[4]

        # 包含所有目标框的最小框到各个边的距离
        d_to_left = x_min
        d_to_right = w - x_max
        d_to_top = y_min
        d_to_bottom = h - y_max

        # 随机扩展这个最小范围
        crop_x_min = int(x_min - random.uniform(0, d_to_left))
        crop_y_min = int(y_min - random.uniform(0, d_to_top))
        crop_x_max = int(x_max + random.uniform(0, d_to_right))
        crop_y_max = int(y_max + random.uniform(0, d_to_bottom))

        # 确保不出界
        crop_x_min = max(0, crop_x_min)
        crop_y_min = max(0, crop_y_min)
        crop_x_max = min(w, crop_x_max)
        crop_y_max = min(h, crop_y_max)

        crop_img = img[crop_y_min:crop_y_max, crop_x_min:crop_x_max]

        # 裁剪bounding boxes
        crop_bboxes = list()
        for bbox in boxes:
            crop_bboxes.append([int(bbox[0] - crop_x_min), int(bbox[1] - crop_y_min),
                                int(bbox[2] - crop_x_min), int(bbox[3] - crop_y_min), bbox[4]])

        return crop_img, crop_bboxes

    # 1 裁切
    def __cropImage2(self, image, boxes):
        ia.seed(1)
        outimage = image
        outboxes = boxes
        for i in range(len(boxes)):
            bbs = BoundingBoxesOnImage([
                BoundingBox(x1=boxes[i][0], x2=boxes[i][2], y1=boxes[i][1], y2=boxes[i][3])
            ], shape=image.shape)
            seq = iaa.Sequential([
                iaa.Crop(percent=(0, 0.4))  # 剪裁
            ], random_order=True)
            image_aug, bbs_aug = seq(image=image, bounding_boxes=bbs)
            outboxes[i][0] = int(bbs_aug[0].x1)
            outboxes[i][2] = int(bbs_aug[0].x2)
            outboxes[i][1] = int(bbs_aug[0].y1)
            outboxes[i][3] = int(bbs_aug[0].y2)
            outimage = image_aug
        return outimage, outboxes

    # 2-平移
    def __translationImage(self, img, boxes):
        """
        平移
        :param img: img
        :param bboxes: bboxes：该图像包含的所有boundingboxes，一个list，每个元素为[x_min,y_min,x_max,y_max]
        :return: shift_img：平移后的图像array；shift_bboxes：平移后的boundingbox的坐标，list
        """

        # 平移图像
        w = img.shape[1]
        h = img.shape[0]

        x_min = w
        x_max = 0
        y_min = h
        y_max = 0

        for bbox in boxes:
            x_min = min(x_min, bbox[0])
            y_min = min(y_min, bbox[1])
            x_max = max(x_max, bbox[2])
            y_max = max(x_max, bbox[3])
            name = bbox[4]

        # 包含所有目标框的最小框到各个边的距离，即每个方向的最大移动距离
        d_to_left = x_min
        d_to_right = w - x_max
        d_to_top = y_min
        d_to_bottom = h - y_max

        # 在矩阵第一行中表示的是[1,0,x],其中x表示图像将向左或向右移动的距离，如果x是正值，则表示向右移动，如果是负值的话，则表示向左移动。
        # 在矩阵第二行表示的是[0,1,y],其中y表示图像将向上或向下移动的距离，如果y是正值的话，则向下移动，如果是负值的话，则向上移动。
        x = random.uniform(-(d_to_left / 3), d_to_right / 3)
        y = random.uniform(-(d_to_top / 3), d_to_bottom / 3)
        M = np.float32([[1, 0, x], [0, 1, y]])

        # 仿射变换
        shift_img = cv2.warpAffine(img, M,
                                   (img.shape[1], img.shape[0]))  # 第一个参数表示我们希望进行变换的图片，第二个参数是我们的平移矩阵，第三个希望展示的结果图片的大小

        # 平移boundingbox
        shift_bboxes = list()
        for bbox in boxes:
            shift_bboxes.append([int(bbox[0] + x), int(bbox[1] + y), int(bbox[2] + x), int(bbox[3] + y), bbox[4]])

        return shift_img, shift_bboxes

    # 3-改变亮度
    def __changeLightofImage(self, img, boxes):
        """
        改变亮度
        :param img: 图像
        :return: img：改变亮度后的图像array
        """
        '''
        adjust_gamma(image, gamma=1, gain=1)函数:
        gamma>1时，输出图像变暗，小于1时，输出图像变亮
        '''
        flag = random.uniform(0.5, 1.5)  ##flag>1为调暗,小于1为调亮
        newBoxes = copy.deepcopy(boxes)
        newImage = exposure.adjust_gamma(img, flag)
        return newImage, newBoxes

    # 4-添加高斯噪声
    def __addNoiseToImage(self, img, boxes):
        """
        加入噪声
        :param img: 图像
        :return: img：加入噪声后的图像array,由于输出的像素是在[0,1]之间,所以得乘以255
        """
        newBoxes = copy.deepcopy(boxes)
        newImage = random_noise(img, mode='gaussian', clip=True) * 255
        return newImage, newBoxes

    # 5-旋转
    def __rotateImage(self, img, boxes):
        """
        旋转
        :param img: 图像
        :param boxes:
        :param angle: 旋转角度
        :param scale: 默认1
        :return: rot_img：旋转后的图像array；rot_bboxes：旋转后的boundingbox坐标list
        """
        '''
        输入:
            img:array,(h,w,c)
            bboxes:该图像包含的所有boundingboxs,一个list,每个元素为[x_min, y_min, x_max, y_max],要确保是数值
            angle:
            scale:默认1
        输出:

        '''
        # 旋转图像
        w = img.shape[1]
        h = img.shape[0]
        angle = random.uniform(-45, 45)
        scale = random.uniform(0.5, 1.5)
        # 角度变弧度
        rangle = np.deg2rad(angle)
        # 计算新图像的宽度和高度，分别为最高点和最低点的垂直距离
        nw = (abs(np.sin(rangle) * h) + abs(np.cos(rangle) * w)) * scale
        nh = (abs(np.cos(rangle) * h) + abs(np.sin(rangle) * w)) * scale
        # 获取图像绕着某一点的旋转矩阵
        # getRotationMatrix2D(Point2f center, double angle, double scale)
        # Point2f center：表示旋转的中心点
        # double angle：表示旋转的角度
        # double scale：图像缩放因子
        rot_mat = cv2.getRotationMatrix2D((nw * 0.5, nh * 0.5), angle, scale)  # 返回 2x3 矩阵
        # 新中心点与旧中心点之间的位置
        rot_move = np.dot(rot_mat, np.array([(nw - w) * 0.5, (nh - h) * 0.5, 0]))
        # the move only affects the translation, so update the translation
        # part of the transform
        rot_mat[0, 2] += rot_move[0]
        rot_mat[1, 2] += rot_move[1]
        # 仿射变换
        rot_img = cv2.warpAffine(img, rot_mat, (int(math.ceil(nw)), int(math.ceil(nh))),
                                 flags=cv2.INTER_LANCZOS4)  # ceil向上取整

        # 矫正boundingbox
        # rot_mat是最终的旋转矩阵
        # 获取原始bbox的四个中点，然后将这四个点转换到旋转后的坐标系下
        rot_bboxes = list()
        for bbox in boxes:
            x_min = bbox[0]
            y_min = bbox[1]
            x_max = bbox[2]
            y_max = bbox[3]
            name = bbox[4]
            point1 = np.dot(rot_mat, np.array([(x_min + x_max) / 2, y_min, 1]))
            point2 = np.dot(rot_mat, np.array([x_max, (y_min + y_max) / 2, 1]))
            point3 = np.dot(rot_mat, np.array([(x_min + x_max) / 2, y_max, 1]))
            point4 = np.dot(rot_mat, np.array([x_min, (y_min + y_max) / 2, 1]))

            # 合并np.array
            concat = np.vstack((point1, point2, point3, point4))  # 在竖直方向上堆叠
            # 改变array类型
            concat = concat.astype(np.int32)
            # 得到旋转后的坐标
            rx, ry, rw, rh = cv2.boundingRect(concat)
            rx_min = rx
            ry_min = ry
            rx_max = rx + rw
            ry_max = ry + rh
            # 加入list中
            rot_bboxes.append([rx_min, ry_min, rx_max, ry_max, name])
        return rot_img, rot_bboxes

    # 6-镜像
    def __flipImage(self, img, bboxes):
        """
        镜像
        :param self:
        :param img:
        :param bboxes:
        :return:
        """
        '''
        镜像后的图片要包含所有的框
        输入：
            img：图像array
            bboxes：该图像包含的所有boundingboxs,一个list,每个元素为[x_min, y_min, x_max, y_max],要确保是数值
        输出:
            flip_img:镜像后的图像array
            flip_bboxes:镜像后的bounding box的坐标list
        '''
        # 镜像图像
        import copy
        flip_img = copy.deepcopy(img)
        if random.random() < 0.5:
            horizon = True
        else:
            horizon = False
        h, w, _ = img.shape
        if horizon:  # 水平翻转
            flip_img = cv2.flip(flip_img, -1)
        else:
            flip_img = cv2.flip(flip_img, 0)
        # ---------------------- 矫正boundingbox ----------------------
        flip_bboxes = list()
        for bbox in bboxes:
            x_min = bbox[0]
            y_min = bbox[1]
            x_max = bbox[2]
            y_max = bbox[3]
            name = bbox[4]
            if horizon:
                flip_bboxes.append([w - x_max, y_min, w - x_min, y_max, name])
            else:
                flip_bboxes.append([x_min, h - y_max, x_max, h - y_min, name])

        return flip_img, flip_bboxes


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Give old image folder.')
    parser.add_argument('--folder',default='E://bug_image_dl/test/all', help='old image folder')
    args = parser.parse_args()
    demo = ImageAugmentation()
    folder = os.listdir(args.folder)
    for filename in folder:
        if os.path.splitext(filename)[1] == '.jpg':  # 目录下包含.json的文件
            name = str(args.folder) + '/' + os.path.splitext(filename)[0]
            demo.changeImages(str(args.folder), "crop", name, 5)
            # demo.changeImages(str(args.folder), "tran", name, 5)
            # demo.changeImages(str(args.folder), "light", name, 5)
            # demo.changeImages(str(args.folder), "noise", name, 5)
            # demo.changeImages(str(args.folder), "rotate", name, 5)
