import shutil

from aip import AipFace
import base64
import cv2
import time
import os

# 登录api
APP_ID = '24767033'
API_KEY = 'Wcz758FYaYUYmedvgOoknhri'
SECRET_KEY = 'zM1izvyVdVG8EtRldvBvzZLMfGwAoaME'
client = AipFace(APP_ID, API_KEY, SECRET_KEY)

def setDir(filepath):
    '''
    如果文件夹不存在就创建，如果文件存在就清空！
    :param filepath:需要创建的文件夹路径
    :return:
    '''
    if not os.path.exists(filepath):
        os.mkdir(filepath)
    else:
        shutil.rmtree(filepath)
        os.mkdir(filepath)
# 查找人脸并剪辑
def face_find(img_path, save_path, star_frame=0, max=100):
    xy_min = 10  # 坐标最小值

    while star_frame < max:  # 最大值
        if os.path.exists(img_path + str(star_frame) + '.jpg') == False:  # 判断文件是否存在
            star_frame += 1
            continue
        with open(img_path + str(star_frame) + '.jpg', 'rb') as f:
            image = f.read()  # 读取一帧
            image = str(base64.b64encode(image), encoding='utf-8')  # 转为base64
            imageType = "BASE64"
            """ 调用人脸检测 """
            options = {"max_face_num": "10"}
            data = client.detect(image, imageType, options)
            # print(data)
            time.sleep(0.5)  # 间隔

            if data["error_code"] == 222202:  # 如果未识别到人脸，即跳转后3帧
                if star_frame + 2 > max:
                    star_frame = max
                else:
                    star_frame += 2
                continue

            json = data["result"]["face_list"]
            j = 1  # 计数现在是第几个人脸
            cap = cv2.imread(img_path + str(star_frame) + '.jpg')  # 创建cv2对象
            while j < len(json) + 1:  # 遍历所有人脸
                x = int(json[j - 1]["location"]["left"])
                y = int(json[j - 1]["location"]["top"])
                h = int(json[j - 1]["location"]["height"])
                w = int(json[j - 1]["location"]["width"])
                # 如果坐标小于被减数即为0，保证剪辑坐标不是负数
                if x < xy_min:
                    x = xy_min
                elif y < xy_min:
                    y = xy_min
                elif h < xy_min:
                    h = xy_min
                elif w < xy_min:
                    w = xy_min
                image = cap[y - xy_min: y + h + xy_min, x - xy_min: x + w + xy_min]  # 剪辑图片
                fe = cv2.imwrite(save_path + str(star_frame) + "_" + str(j) + '.jpg', image)  # 保存图片
                #if fe:
                   # print("成功提取" + str(star_frame) + '.jpg 的第 ' + str(j) + " 张人脸")
                j += 1
            star_frame += 1  # 识别到第几张图片


# 按帧截取图片
def get_img(mp4_path, save_path, rate=1000):
    cap = cv2.VideoCapture(mp4_path)  # 创建一个 VideoCapture 对象
    suc = cap.isOpened()  # 是否成功打开
    frame_count = 0
    out_count = 0
    while suc:
        if out_count > 10000:  # 最多取出多少张
            break
        out_count += 1
        suc, frame = cap.read()
        cap.set(cv2.CAP_PROP_POS_MSEC, rate * out_count)  # 设置视频时间
        if suc == False:
            break
        print(save_path + str(int(frame_count * rate / 1000)) + ".jpg")
        print(cv2.imwrite(save_path + str(int(frame_count * rate / 1000)) + ".jpg", frame))
        frame_count += 1


