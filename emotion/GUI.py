import tkinter
import forall
from tkinter import *
from PIL import Image, ImageTk
top = tkinter.Tk()
#face=forall.face()
voice=forall.voice()
text=forall.text()
top.geometry("1200x750+200+20")
label1=tkinter.Label(top,text="欢迎来到居家情绪检测系统",font = ('Arial' , 25))
canvas = tkinter.Canvas(top, bg='white',bd='5',
                            height=375, width=600)
# 进入消息循环
img = Image.open('images/result/l.png')
# img=img.sample('100*100')
img = img.resize((600, 375), resample=0)
img = ImageTk.PhotoImage(img)
# print(img_png.size)
image = canvas.create_image(300, 188, anchor=CENTER, image=img)
label1.pack()

def run():
    max=forall.face()
    label2 = tkinter.Label(top, text="该时段表情预测结果为：" + max, font=('Arial', 15))
    label3 = tkinter.Label(top, text="该时段语音预测结果为：" + voice, font=('Arial', 15))
    label4 = tkinter.Label(top, text="该时段文本预测结果为：" + text, font=('Arial', 15))
    label2.pack()
    label3.pack()
    label4.pack()
def imgshow():
        canvas.pack()
B1 = tkinter.Button(top, text="开始监测",font='20',command=lambda :run())
B2 = tkinter.Button(top, text="点击查看表情结果展示",font='20',command=lambda :imgshow())
B1.pack()
B2.pack()
top.mainloop()

