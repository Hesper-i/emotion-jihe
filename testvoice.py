# coding:utf-8
import os

from pydub import AudioSegment
from pydub.silence import split_on_silence

sound = AudioSegment.from_mp3("source/fyh_happy.wav")
loudness = sound.dBFS
chunks = split_on_silence(sound,
                          min_silence_len=430,

                          silence_thresh=-45,
                          keep_silence=400
                          )
print('总分段：', len(chunks))
# 放弃长度小于2秒的录音片段
for i in list(range(len(chunks)))[::-1]:
    if len(chunks[i]) <= 2000 or len(chunks[i]) >= 10000000:
        chunks.pop(i)
print('取有效分段(大于2s小于10s)：', len(chunks))
for i, chunk in enumerate(chunks):
    chunk.export("voice/{0}.wav".format(i), format="wav")

