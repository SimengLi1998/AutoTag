#!/usr/bin/python
# -*- coding: UTF-8 -*-

from flask import Flask, request

app=Flask(__name__)
 
import numpy as np
import pandas as pd
import Levenshtein as ln
from sentence_transformers import SentenceTransformer

import copy
# import multiprocessing as mp
import torch.multiprocessing as mp #使用pytorch cuda多线程支持
import torch
# import os

# os.environ['TOKENIZERS_PARALLELISM']="false"
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1" #关闭CUDA加速
# torch.backends.cudnn.enabled = False

#全局变量保存checklist
checklist = []
checklist_embeddings = []
model = None
word_path = 'clist.csv'
embedding_path = 'embeddings.csv'
cores = mp.cpu_count() #获取本机CPU核数


def read_checklist():
    df = pd.read_csv(word_path,  
    header = 0, #None表示没有， 0表示第1行
    encoding='gbk', #注意编码格式
    names=['name'], 
    index_col=None, #没有index列
    delimiter=",")
    #获取到list
    tmp = list(df.name)
    #去掉大小写
    tmp = [i.lower() for i in tmp]
    return tmp

def parse_list(str):
    #解析数组字符串
    li = np.array([float(i) for i in list(str.replace('[','').replace(']','').split(','))])
    return li

def read_embedding():
    print(f"读取保存的embeddings")
    df = pd.read_csv(embedding_path,  
    header = 0, #None表示没有， 0表示第1行
    encoding='gbk',
    names=['word', 'embedding'],
    dtype={"word": str, "embedding": str},  #按指定类型读取列数据
    index_col=None, #没有index列
    delimiter=",")
    print(f"读取保存的embeddings完毕！")
    words_embeddings = zip(df.embedding, list(df.word))
    return words_embeddings
    
#结构相似函数
def wordSimilar(str2, checklist):
    #当str2不为空时开始检测
    if str2 is not None:
        
        str2 = str2.lower()
        result =[]
        
        for str in checklist:
            words = str.split(" ")
            name_len = len(words)

            if name_len > 1:
                ss = ln.ratio(str, str2)
                jss = ln.jaro(str, str2)
                jwss = ln.jaro_winkler(str, str2)
                if ss >=0.80:
                    result.append({"type":"ss", "word":str, "probability": ss})
                if jss >=0.85:
                    result.append({"type":"jss", "word":str, "probability":jss})
                if jwss >=0.85:
                    result.append({"type":"jwss", "word":str, "probability":jwss})
                samewords = [i for i in words if str2.find(i)!=-1]
                if len(samewords) == len(str.split(" ")):
                    result.append({"type":"swl", "word":str, "probability":1})
            else:
                if str[0] == str2[0]:
                    ss = ln.ratio(str, str2)
                    jss = ln.jaro(str, str2)
                    jwss = ln.jaro_winkler(str, str2)
                    if ss >=0.80:
                        result.append({"type":"ss", "word":str, "probability": ss})
                    if jss >=0.85:
                        result.append({"type":"jss", "word":str, "probability":jss})
                    if jwss >=0.85:
                        result.append({"type":"jwss", "word":str, "probability":jwss})
                else:
                    continue
                    # print(f"单个词首字母不匹配")
        
        # print(high_score)
    return result

#计算余弦夹角
def cosine_similarity(x,y):
    num = x.dot(y.T)
    denom = np.linalg.norm(x) * np.linalg.norm(y)
    return num / denom

#语义相似函数
def meanSimilar(model, checklist_embeddings, str2):
    
    vec2 = model.encode([str2])[0]
    #print(f'str2: {str2}' )
    result = []

    for row in checklist_embeddings:
        
        cos_score = cosine_similarity(parse_list(row[0]), vec2)
                
        #定义语义相似度阈值
        threshold_cos = 0.66
        # cos_score = math.cos(distance_cos)
        if cos_score >= threshold_cos:
            result.append({"type":"ms", "word":row[1], "probability":cos_score})
            # result.append((row[1], cos_score))
            # print(f"与{row[1]}余弦角: {math.cos(distance_cos)}")
    return result    
        

def getModel(checklist):
    
    #加载在线模型
    # model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    # model.share_memory()
    #加载huggingface本地模型文件
    # tokenizer = AutoTokenizer.from_pretrained('/home/dbt/project_pro/brandcheck/all-MiniLM-L6-v2')
    # model = AutoModel.from_pretrained('/home/dbt/project_pro/brandcheck/all-MiniLM-L6-v2')
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    embeddings = model.encode(checklist).tolist()
    # for i in embeddings:
    #     print(i)
    #     break
    # checklist_embeddings = zip(embeddings, checklist)
    df = pd.DataFrame({'words': checklist,
                   'embeddings': embeddings
                   })
    df.to_csv(embedding_path,index=False)
    return model

def Thandler(word, checklist, checklist_embeddings, model):
    tmp = []
    # result = []

    tmp.extend(wordSimilar(word, checklist))


    embedding_copy = copy.deepcopy(checklist_embeddings)

    data = meanSimilar(model, embedding_copy, word)


    tmp.extend(data)

    tmp2 = {"word":word, "value":tmp}

    return tmp2

@app.route('/brandcheck', methods=['GET', 'POST'])
def get_json():
    checklist_embeddings = read_embedding()
    #2.传入 json对象
    params = request.json
    #增加多线程

    brand_words = params['brand']
    ctx = torch.multiprocessing.get_context("spawn")  #使用cuda加速此句

       
    pool = ctx.Pool(processes=cores-1)
    # pool = mp.Pool(processes=cores-1) #不使用CUDA加速时
    results = [] #保存各个进程执行结果
    tmp = []
    if len(brand_words)>0:
        for word in brand_words:
            # print(f"word: {word}, wordtype: {type(word)}")
            tasks = pool.apply_async(Thandler, args=(word, checklist, checklist_embeddings, model), callback=tmp.append)
            # tasks.start()
            print(f"Starting tasks...{word}")
    tasks.wait()
    pool.close()  #必须先close()才能调用join(),不然报错
    pool.join()   #调用join之前，先调用close函数，否则会出错。执行完close后不会有新的进程加入到pool,join函数等待所有子进程结束
    results = [i for i in tmp]
    # #print(f"results: {results}")
    # # return json.dumps({"data": results}, cls=MyEncoder)
    
    return {"data": results}




if __name__=='__main__':
    app.debug=True
    print('加载商标列表...')
    checklist = read_checklist()
    
    print('加载商标列表完毕！')
    print('生成商标列表embeddings')
    model = getModel(checklist)
    print('生成商标列表embeddings完毕！')

    
    # app.run(host='127.0.0.1',port=7777)
    app.run(host='10.105.64.98',port=7777)
    # app.run(host='10.105.128.2',port=5000)
    

