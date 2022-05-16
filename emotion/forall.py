from mp4towav import  *
#from text2emo import *
import text2emo as tex
from voice2text import voice2text_main
from emotion_recognition import EmotionRecognizer
from sklearn.svm import SVC
import face.transforms as transforms
from face.models import *
from tkinter import *
import tkinter as tk
import imageio
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image,ImageTk
import torch
import torch.nn.functional as F
import os
from torch.autograd import Variable
from skimage import io
from skimage.transform import resize
from pydub import AudioSegment
from pydub.silence import split_on_silence
import process as p

resultface=''
resultvoice=''
resulttext=''

cut_size = 44

transform_test = transforms.Compose([
    transforms.TenCrop(cut_size),
    transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
])

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

def findAllFile(base):
    for root,ds, fs in os.walk(base):
        for f in fs:
            yield f


def face(audio_path):
    p.setDir("photo")
    p.setDir("set")
    p.get_img(audio_path+'.mp4', "photo/", 1000)
    p.face_find("photo/", "set/")
    positive_num=0
    positive_sum=0
    natural_num=0
    natural_sum=0
    negative_num=0
    negative_sum=0
    a = []
    angry = 0
    disgust = 0
    fear = 0
    happy = 0
    sad = 0
    surp = 0
    neu = 0
    for imgs in findAllFile('set'):
        imgpath=os.path.join('set/%s'%imgs)
        raw_img = io.imread(imgpath)
        gray = rgb2gray(raw_img)
        gray = resize(gray, (48,48), mode='symmetric').astype(np.uint8)
        img = gray[:, :, np.newaxis]
        img = np.concatenate((img, img, img), axis=2)
        img = Image.fromarray(img)
        inputs = transform_test(img)
        #情绪对应值
        class_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
        net = VGG('VGG19')
        checkpoint = torch.load(os.path.join('face/FER2013_VGG19', 'PrivateTest_model.t7'),map_location='cpu')
        net.load_state_dict(checkpoint['net'])
        #net.cuda()
        net.eval()

        ncrops, c, h, w = np.shape(inputs)

        inputs = inputs.view(-1, c, h, w)
        #inputs = inputs.cuda()
        inputs = Variable(inputs, volatile=True)
        outputs = net(inputs)

        outputs_avg = outputs.view(ncrops, -1).mean(0)  # avg over crops
        score = F.softmax(outputs_avg)
        #这个是每张图片概率值
        #print(score.data)
        face_pre=score.data.numpy()
        #print(face_pre)
        _, predicted = torch.max(outputs_avg.data, 0)

        '''curr_time = datetime.datetime.now()
        timestamp = datetime.datetime.strftime(curr_time, '%Y-%m-%d %H:%M:%S')'''
        #这个是每张图片预测结果
        idex=predicted.cpu().numpy()
        print("表情心情： %s" %str(class_names[int(idex)]))
        print(face_pre[idex])

        if(str(class_names[int(idex)])=='Angry'):
            a.append(-1)
            angry=angry+1
            negative_num = negative_num + 1
            negative_sum+=face_pre[idex]

        elif(str(class_names[int(idex)])=='Disgust'):
            a.append(-1)
            disgust=disgust+1
            negative_num = negative_num + 1
            negative_sum += face_pre[idex]

        elif (str(class_names[int(idex)]) == 'Fear'):
            negative_num=negative_num+1
            negative_sum += face_pre[idex]
            a.append(-1)
            fear=fear+1
        elif (str(class_names[int(idex)]) == 'Happy'):
            positive_num=positive_num+1
            positive_sum+=face_pre[idex]
            a.append(1)
            happy=happy+1
        elif (str(class_names[int(idex)]) == 'Sad'):
            negative_num=negative_num+1
            negative_sum += face_pre[idex]
            a.append(-1)
            sad=sad+1
        elif (str(class_names[int(idex)]) == 'Surprise'):
            positive_num=positive_num+1
            positive_sum += face_pre[idex]
            a.append(1)
            surp=surp+1
        elif (str(class_names[int(idex)]) == 'Neutral'):
            natural_num=natural_num+1
            natural_sum+=face_pre[idex]
            a.append(0)
            neu=neu+1

    #print(a)
    countemo=[angry,disgust,fear,happy,sad,surp,neu]
    counts=[positive_num,natural_num,negative_num]
    #print(counts)
    maxface=max(countemo,key=abs)
    index=countemo.index(maxface)
    global resultface
    resultface = class_names[index]
    counts_pre={'positive':0,'neutral':0,'negative':0}
    Max = max(counts,key=abs)
    idex_counts = counts.index(Max)
    #print(counts)
    #print(idex_counts)

    for i in range(0,3):
        if counts[i]==0:
            counts[i]=1
    sum=natural_sum+positive_sum+negative_sum
    counts_pre['neutral'] = natural_sum /sum
    counts_pre['positive'] = positive_sum /sum
    counts_pre['negative'] = negative_sum /sum
    if idex_counts==0:
        faces_emo='positive'

    elif idex_counts==1:
        faces_emo = 'neutral'

    else:
        faces_emo = 'negative'
    print(faces_emo)
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.figure(figsize=(5,2))
    plt.title("表情预测结果", fontsize=10)
    x = range(len(os.listdir('set')))
    y = range(-1, 2, 1)
    x_ticks_label = ["{}".format(i + 1) for i in x]
    plt.plot(x, a, 'r*--', alpha=0.5, linewidth=1, label='1为正向，0为中性，-1为负向')  # 'bo-'表示蓝色实线，数据点实心原点标注
    ## plot中参数的含义分别是横轴值，纵轴值，线的形状（'s'方块,'o'实心圆点，'*'五角星   ...，颜色，透明度,线的宽度和标签 ，
    plt.xticks(x,x_ticks_label, rotation=90)
    plt.yticks(y)
    plt.legend()  # 显示上面的label
    plt.xlabel('time')  # x_label
    plt.ylabel('emo')  # y_label
    plt.savefig(os.path.join('results/face.png'))
    plt.close()
    #print(counts_pre)
    #print(faces_emo)
    return counts_pre

def findAllFile(base):
    for root,ds, fs in os.walk(base):
        for f in fs:
            yield f

def voice(audio_path):
    # 处理视频转语音文本
    mp4_path = os.getcwd() + audio_path+'.mp4'
    wav_path = os.getcwd() + audio_path+'.wav'
    sampling_rate = 16000
    mp4_to_wav(mp4_path, wav_path, sampling_rate)
    voice=[]
    emo = ['negative', 'neutral', 'positive']
    sad = 0
    neu = 0
    happy = 0
    my_model = SVC(probability=True)
    rec = EmotionRecognizer(model=my_model, emotions=['sad', 'neutral', 'happy'], balance=True, verbose=0)
    rec.train()
    #print(rec.predict(wav_path))
    #print(rec.predict_proba(wav_path))
    rec.determine_best_model()
    # get the determined sklearn model name
    print(rec.model.__class__.__name__, "is the best")
    # get the test accuracy score for the best estimator
    print("Test score:", rec.test_score())
    # check the train accuracy for that model
    print("Train score:", rec.train_score())
    global resultvoice
    resultvoice=rec.predict(audio_path+'.wav')
    resultvoice1=rec.predict_proba(audio_path+'.wav')
    value_resultvoice1=resultvoice1.values()
    if(resultvoice=='sad'):
        sad=sad+1
        '''voice.append(maxresultvoice1)
        voice.append(0)
        voice.append(0)'''
    elif(resultvoice=='neutral'):
        '''voice.append(0)
        voice.append(maxresultvoice1)
        voice.append(0)'''
        neu=neu+1
    elif(resultvoice=='happy'):
        '''voice.append(0)
        voice.append(0)
        voice.append(maxresultvoice1)'''
        happy=happy+1
    countvoice = [sad, neu, happy]
    maxvoice = max(countvoice, key=abs)
    index = countvoice.index(maxvoice)
    resultvoice= emo[index]
    print(voice)
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    emo = ['正向', '中性', '负向']
    emo.reverse()
    #data1 = [1, 1, 1]
    # data2 = [0, 2, 0]
    color=['orange','#00bfff','#90ee90']
    fig, ax = plt.subplots(figsize=(5, 2.3))
   # b = ax.barh(range(len(emo)), data1, color='white')
    b = ax.barh(range(len(emo)), value_resultvoice1, color=color,height=0.3)
    # 添加数据标签
    for rect in b:
        w = rect.get_width()
        ax.text(w, rect.get_y() + rect.get_height() / 2, '%f' %float(w), ha='left', va='center')
    # 设置y轴纵坐标上的刻度线标签。
    ax.set_yticks(range(len(emo)))
    ax.set_yticklabels(emo)
    # 不要x横坐标上的label标签。
    plt.xticks(range(2))
    plt.title("语音预测结果", fontsize=10)
    plt.savefig(os.path.join('results/voice.png'))
    plt.close()
    return rec.predict_proba(audio_path+'.wav')


# 文本
def text():
    emo=['negative','neutral','positive']
    neg=0
    neu=0
    pos=0
    text = []
    filepath =os.path.join(audio_path+'.wav')
    voice2text_main(filepath)
    with open("text/result.txt") as file:
        input = file.read()
    str = tex.get_emotion(input)
    textresult=str[0]['label']
    text_pres = {'positive': 0, 'neutral': 0, 'negative': 0}
    for i in range(0, 3):
        if str[i]['label'] == 'neutral':
            text_pres['neutral'] = str[i]['prob']
        elif str[i]['label'] == 'pessimistic':
            text_pres['negative'] = str[i]['prob']
        else:
            text_pres['positive'] = str[i]['prob']
    value_text_pres=text_pres.values()
    if(textresult=='negative'):
        '''text.append(maxtext_pres)
        text.append(0)
        text.append(0)'''
        neg=neg+1
    elif(textresult=='neutral'):
        '''text.append(0)
        text.append(maxtext_pres)
        text.append(0)'''
        neu=neu+1
    elif(textresult=='pessimistic'):

        pos=pos+1


    #print(text_pres)
    print(text)
    counttext=[neg,neu,pos]
    maxtext=max(counttext, key=abs)
    index=counttext.index(maxtext)
    global resulttext
    resulttext=emo[index]
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    emo = ['正向', '中性', '负向']
    emo.reverse()
    #data1 = [1, 1, 1]
    #data2 = [0, 2, 0]
    color=['orange','#00bfff','#90ee90']
    fig, ax = plt.subplots(figsize=(5,2.3))
    #b = ax.barh(range(len(emo)), data1, color='white')
    b = ax.barh(range(len(emo)), value_text_pres, color=color,height=0.3)
    # 添加数据标签
    for rect in b:
        w = rect.get_width()
        ax.text(w, rect.get_y() + rect.get_height() / 2, '%f' % float(w), ha='left', va='center')
    # 设置y轴纵坐标上的刻度线标签。
    ax.set_yticks(range(len(emo)))
    ax.set_yticklabels(emo)
    # 不要x横坐标上的label标签。
    plt.xticks(range(2))
    plt.title("文本预测结果", fontsize=10)
    plt.savefig(os.path.join('results/text.png'))
    plt.close()
    return text_pres


if __name__ == '__main__':
    audio_path = './source/laoren2'
    ans_dict={'正向':0,'中性':0,'负向':0}
    face_dict=face(audio_path)
    voice_dict=voice(audio_path)
    text_dict=text()
    ans_dict['正向']=face_dict['positive']*0.6+voice_dict['happy']*0.2+text_dict['positive']*0.2
    ans_dict['中性'] = face_dict['neutral']*0.6+voice_dict['neutral']*0.2+text_dict['neutral']*0.2
    ans_dict['负向'] =face_dict['negative']*0.6+voice_dict['sad']*0.2+text_dict['negative']*0.2
    finallresult = max(ans_dict, key=lambda x: ans_dict[x])
    print(face_dict)
    print(voice_dict)
    print(text_dict)
    print(ans_dict)
    print(finallresult)


    def stream():
        try:
            image = video.get_next_data()
            frame_image = Image.fromarray(image)
            frame_image = ImageTk.PhotoImage(frame_image)
            l1.config(image=frame_image)
            l1.image = frame_image
            l1.after(delay, lambda: stream())
        except:
            video.close()
            return


    top = tk.Tk()
    top.geometry("1000x720+1+1")
    top.title = '情绪检测系统'
    fresult = tk.Frame(top)
    fimg = tk.Frame(top)
    fvideo = tk.Frame(top)
    fimg.pack(side='left', anchor='nw')
    fvideo.pack(side='top', anchor='ne')
    fresult.pack(side='top', anchor='sw')

    canvas1 = tk.Canvas(fimg , bd='5',
                        height=250, width=500)
    canvas2 = tk.Canvas(fimg, bd='5',
                        height=230, width=500)
    canvas3 = tk.Canvas(fimg, bd='5',
                        height=230, width=500)
    img1 = Image.open('results/face.png')
    # img=img.sample('100*100')
    img1 = img1.resize((500, 250), resample=0)
    img1.save('results/face.png')
    img1 = ImageTk.PhotoImage(img1)
    # print(img_png.size)
    image1 = canvas1.create_image(250, 125, image=img1)

    img2 = Image.open('results/voice.png')
    # img=img.sample('100*100')
    img2 = img2.resize((500, 230), resample=0)
    img2.save('results/voice.png')
    img2 = ImageTk.PhotoImage(img2)
    image2 = canvas2.create_image(250, 115, image=img2)

    img3 = Image.open('results/text.png')
    # img=img.sample('100*100')
    img3 = img3.resize((500, 230), resample=0)
    img3.save('results/text.png')
    img3 = ImageTk.PhotoImage(img3)
    # print(img_png.size)
    image3 = canvas3.create_image(250, 115, image=img3)
    canvas1.pack(side='top', anchor='w')
    canvas2.pack(side='top', anchor='w')
    canvas3.pack(side='top', anchor='w')
    if (finallresult == '正向'):
        color = '#90ee90'
    elif (finallresult == '负向'):
        color = 'orange'
    else:
        color = '#00bfff'
    label2 = tk.Label(fresult, text="正向" ,bg='#90ee90',width=10 ,font=('Arial', 15))
    label22 = tk.Label(fresult, text='', width=3)
    label6 = tk.Label(fresult, text='\n\n', width=30)
    label3 = tk.Label(fresult, text="中性",bg='#00bfff',width=10, font=('Arial', 15))
    label33 = tk.Label(fresult, text='', width=3)
    label4 = tk.Label(fresult, text="负向",bg='orange' ,width=10,font=('Arial', 15))
    label44 = tk.Label(fresult, text='', width=3)
    label5 = tk.Label(fresult, text="综合预测结果为：" + finallresult,bg=color,font=('Arial', 20))
    label7 = tk.Label(fresult, text='\n', width=30)
    label7.pack(side='top', anchor='w')
    label5.pack(side='top', anchor='center')
    label6.pack(side='top', anchor='w')
    label2.pack(side='right',anchor='center')
    label22.pack(side='right', anchor='center')
    label3.pack(side='right',anchor='center')
    label33.pack(side='right', anchor='center')
    label4.pack(side='right',anchor='center')
    label44.pack(side='right', anchor='center')

    l1 = Label(fvideo)
    video = imageio.get_reader(audio_path+'.mp4')
    delay = int(1000 / video.get_meta_data()['fps'])
    try:
        image = video.get_next_data()
        frame_image = Image.fromarray(image)
        frame_image = ImageTk.PhotoImage(frame_image)
        l1.config(image=frame_image)
        l1.image = frame_image
        l1.after(delay, lambda: stream())
    except:
        video.close()
    l1.pack(side='top', anchor='ne')
    top.mainloop()