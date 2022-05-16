# -*- coding: utf-8 -*-
#!/usr/bin/env python

import urllib
import urllib.request
import json
from voice2text import voice2text_main
import pandas
#client_id 为官网获取的AK， client_secret 为官网获取的SK
client_id ='InUuXAjyxl24b1zCNIKZVxyB'
client_secret ='p8mQGFyQVkyE9vLZcKcHqjK4GyCO8Hve'

#获取token
def get_token():
    host = 'https://aip.baidubce.com/oauth/2.0/token?grant_type=client_credentials&client_id=' + client_id + '&client_secret=' + client_secret
    request = urllib.request.Request(host)
    request.add_header('Content-Type', 'application/json; charset=UTF-8')
    response = urllib.request.urlopen(request)
    token_content = response.read()
    if token_content:
        token_info = json.loads(token_content)
        token_key = token_info['access_token']
    return token_key
def get_emotion(content):
    token=get_token()
    url = 'https://aip.baidubce.com/rpc/2.0/nlp/v1/emotion'
    params = dict()
    params['scene'] = 'talk'
    params['text'] = content
    params = json.dumps(params).encode('utf-8')
    access_token = token
    url = url + "?access_token=" + access_token
    url = url + "&charset=UTF-8" # 此处指定输入文本为UTF-8编码，返回编码也为UTF-8
    request = urllib.request.Request(url=url, data=params)
    request.add_header('Content-Type', 'application/json')
    response = urllib.request.urlopen(request)
    content = response.read()
    if content:
        content=content.decode('utf-8')
        data = json.loads(content)
        result = data['items'][0]['label']

        if result == 'neutral':
            return data['items']
        else:
            return data['items']
    else:
        return ''
if __name__ == '__main__':
    filepath="./source/fyh_happy.wav"
    voice2text_main(filepath)
    with open("text/result.txt") as file:
        input = file.read()
    str=get_emotion(input)
    print('the emotion and precision are :')
    print(str)
    print(str[0]['label'])
    print(str[0]['prob'])