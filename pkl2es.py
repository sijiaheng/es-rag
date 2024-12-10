#读取pkl数据写入es数据库中，需要指定pkl的路径
import os
import pandas as pd
import re
import transformers
from text2vec import SentenceModel
from elasticsearch_dsl import Document, Text, Index, connections,DenseVector,Integer
import pickle
import warnings
from tqdm import tqdm

warnings.filterwarnings('ignore')



pkl_path = "/data/ES_RAG/pkl"
#与es建立连接
# client = connections.create_connection(hosts=['https://10.1.48.56:9200'], timeout=20,http_auth=('elastic','elastic'),verify_certs=False)
client = connections.create_connection(hosts = ['http://localhost:9200'],timeout = 20)

#指定索引
index_name = "knowledge_test"#索引名字
index = Index(index_name)
if index.exists():
    index.delete()

class VecDocument(Document):#存储的格式与索引
    filename = Text()
    num = Integer()
    text = Text()
    vector = DenseVector(dims = 1024)

    class Index:
        name = index_name
#创建索引并将映射与索引关联
VecDocument.init()

def upload_es(pkl_path):
    # 建立连接
    for filename in os.listdir(pkl_path):#查看pkl文件夹中的所有文件
        if filename.endswith(".pkl"):
            now_pkl_path = os.path.join(pkl_path, filename)
            #读取now_pkl_path指定的pkl文件，对其操作
            # 读取 pkl 文件
            with open(now_pkl_path, 'rb') as f:
                data = pickle.load(f)
                for row in tqdm(data, desc=f"Uploading documents of {filename}", unit="document"):
                    knowledge_document = VecDocument(filename = row['filename'],
                                         num = row['num'],
                                         text = row['text'],
                                         vector = row['embedding']
                                         )
                    knowledge_document.save(using=client)

upload_es(pkl_path)