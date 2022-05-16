# -*- coding: utf-8 -*-
import os, sys
import subprocess
import ffmpeg
from scipy.io import wavfile

def ffmpeg_VideoToAudio(VideoPath, WavPath):
    # 提取视频路径下所有文件名
    videos = os.listdir(VideoPath)
    count = 0
    for video in videos:
        # 提取视频的全路径名（含路径+文件名）
        video_path = VideoPath + "\\" + video
        # 合成输出音频的全路径名（不含后缀）
        wav_path = WavPath + "\\" + os.path.splitext(video)[0]
        #print("output: "+ wav_path)
        # 提取视频中的音频信息
        strcmd = "ffmpeg -i " + video_path + " -f wav " + wav_path + ".wav"
        print(strcmd)
        subprocess.run(strcmd, shell=True, encoding='utf-8', check=True)
        #s=os.system(strcmd)
        #subprocess.getoutput(strcmd)



VideoPath = r'.\surprise'
WavPath = r'.\surprise'
ffmpeg_VideoToAudio(VideoPath,WavPath)

