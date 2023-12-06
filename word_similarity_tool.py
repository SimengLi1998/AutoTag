import tkinter as tk
from tkinter.filedialog import *
from tkinter import filedialog
import  pandas as pd
import requests 
import json 
import os

import logging
def setup_logger(logger_name, log_file, level=logging.INFO):
    l = logging.getLogger(logger_name)
    formatter = logging.Formatter('%(asctime)s : %(message)s')
    fileHandler = logging.FileHandler(log_file, mode='w')
    fileHandler.setFormatter(formatter)
    streamHandler = logging.StreamHandler()
    streamHandler.setFormatter(formatter)
 
    l.setLevel(level)
    l.addHandler(fileHandler)
    l.addHandler(streamHandler)


root = tk.Tk()
root.title("语义相似度计算")
# root.geometry('400x200')
filename_1 = tk.StringVar()
global url
url = tk.StringVar()
global Top_N_para

Top_N_para = tk.IntVar()
Top_N_para.set(2)
url.set("http://10.105.64.98:7788/brandcheck")

def openFile_1():
    filepath = askopenfilename()  # 选择打开什么文件，返回文件名
    if filepath.strip() != '':
        filename_1.set(filepath)  # 设置变量filename的值
    else:
        print("do not choose file")



# def openDir():
#     fileDir = askdirectory()  # 选择目录，返回目录名
#     if fileDir.strip() != '':
#         dirpath.set(fileDir)  # 设置变量outputpath的值
#     else:
#         print("do not choose Dir")

# def fileSave():
#     filenewpath = asksaveasfilename(defaultextension='.csv')  # 设置保存文件，并返回文件名，指定文件名后缀为.txt
#     if filenewpath.strip() != '':
#         filenewname.set(filenewpath)  # 设置变量filenewname的值
#     else:
#         print("do not save file")

setup_logger('word_similarity', r'/ %s A_word_similarity.log'%(os.path.split(filename_1.get())[0]))#.format(os.path.split(filename_1.get())[0])
log1 = logging.getLogger('word_similarity')

def post_request():
    log1.info('Calculating ~~~')
    print(filename_1.get())

    data = []
    with open(filename_1.get(), "r") as f:
        for line in f.readlines():
            line = line.strip('\n')  #去掉列表中每一个元素的换行符
            data.append(line)
    print(data)

    # data = pd.read_csv(filename_1.get(),dtype={'brand': str})
    # print(data['brand'].tolist())
    data_post =  {'brand':data,"Top_N":Top_N_para.get()}
    print(data_post)
    URL = url.get() #"http://10.105.64.98:7788/brandcheck" 
    # 2-json参数会自动将字典类型的对象转换为json格式 
    result = requests.post(URL, json=data_post)
    result_fin = result.text
    print(result_fin)

    df = pd.DataFrame(columns=["word_input","probability", "type","word_output"])
    for i,row in enumerate(eval(result_fin).get('data')):
        df.loc[i,'word_input'] = row.get('word').lower()
        df.loc[i,'probability'] = row.get('value').get('probability')
        df.loc[i,'type'] = row.get('value').get('type')
        df.loc[i,'word_output'] = row.get('value').get('word')
    print(os.path.split(filename_1.get()))
    df.to_csv(os.path.split(filename_1.get())[0]+'/output_result.csv',index=False)
    log1.info('Input Data:{}'.format(data))
    log1.info("Result Data:{}".format(result_fin))
    log1.info('Calculating End !!!')


# 清空文本框
def clear():
    Top_N_para.set('')
 


# 布局控件
tk.Label(root, text="Word Similarity Calculator",font=('Times Newer Roman',14)).grid(row=0, column=1, padx=5, pady=5)
# 打开文件

tk.Label(root, text='选择文件').grid(row=1, column=0, padx=5, pady=5)
tk.Entry(root, textvariable=filename_1,width=40).grid(row=1, column=1, padx=5, pady=5)
tk.Button(root, text='打开文件', command=openFile_1).grid(row=1, column=2, padx=5, pady=5)



# 保存文件
tk.Label(root, text='设置参数').grid(row=3, column=0, padx=5, pady=5)
tk.Entry(root, textvariable=Top_N_para,width=40).grid(row=3, column=1, padx=5, pady=5)
tk.Button(root, text='点击清除', command=clear,height=1).grid(row=3, column=2, padx=5, pady=5)

# 设置地址
tk.Label(root, text='设置地址').grid(row=5, column=0, padx=5, pady=5)
tk.Entry(root, textvariable=url,width=40).grid(row=5, column=1, padx=5, pady=5)
tk.Button(root, text='点击计算', command=post_request).grid(row=5, column=2, padx=5, pady=5)




root.mainloop()
