#coding=gbk
import tkinter as tk
from tkinter import *
from PIL import Image, ImageTk
import imageio

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
top.geometry("1200x800+200+20")
top.title = '情绪检测系统'
fresult = tk.Frame(top)
fimg = tk.Frame(top)
fvideo = tk.Frame(top)
fimg.pack(side='left', anchor='nw')
fvideo.pack(side='top', anchor='ne')
fresult.pack(side='top', anchor='se')

canvas1 = tk.Canvas(fimg, bg='blue', bd='5',
                    height=375, width=600)
canvas2 = tk.Canvas(fimg, bg='blue', bd='5',
                    height=200, width=600)
canvas3 = tk.Canvas(fimg, bg='blue', bd='5',
                    height=200, width=600)
img1 = Image.open('results/face.png')
# img=img.sample('100*100')
img1 = img1.resize((600, 375), resample=0)
img1.save('results/face.png')
img1 = ImageTk.PhotoImage(img1)
# print(img_png.size)
image1 = canvas1.create_image(300, 188, image=img1)

img2 = Image.open('results/voice.png')
# img=img.sample('100*100')
img2 = img2.resize((600, 200), resample=0)
img2.save('results/voice.png')
img2 = ImageTk.PhotoImage(img2)
image2 = canvas2.create_image(300, 100, image=img2)

img3 = Image.open('results/text.png')
# img=img.sample('100*100')
img3 = img3.resize((600, 200), resample=0)
img3.save('results/text.png')
img3 = ImageTk.PhotoImage(img3)
# print(img_png.size)
image3 = canvas3.create_image(300, 100, image=img3)
canvas1.pack(side='top', anchor='w')
canvas2.pack(side='top', anchor='w')
canvas3.pack(side='top', anchor='w')
max = "happy"
label2 = tk.Label(fresult, text="该时段表情预测结果为：" + max, font=('Arial', 15))
label3 = tk.Label(fresult, text="该时段语音预测结果为：" + max, font=('Arial', 15))
label4 = tk.Label(fresult, text="该时段文本预测结果为：" + max, font=('Arial', 15))
label2.pack(side='top', anchor='w')
label3.pack(side='top', anchor='w')
label4.pack(side='top', anchor='w')
l1 = Label(fvideo)
video = imageio.get_reader("source/4.mp4")
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