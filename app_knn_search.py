#主要是为了进行向量搜索，返回匹配的top-n的知识
#采用近似knn搜索
import pandas as pd
import numpy as np
import transformers
from text2vec import SentenceModel
import torch.nn.functional as F
import numpy as np
from elasticsearch_dsl import Search, Document, Index, connections, DenseVector
from elasticsearch_dsl.query import ScriptScore, Q, MatchAll
from elasticsearch import Elasticsearch
import time
import warnings

warnings.filterwarnings('ignore')#忽略警告

index_name = "testnorm"#索引名字

es = Elasticsearch(
    hosts=['https://localhost:9200'],  # 替换为您的 Elasticsearch 集群地址
    http_auth=('elastic', 'elastic'),
    # 如果您的 Elasticsearch 集群使用自签名证书，请将 verify_certs 设置为 False
    verify_certs=False,
    timeout=60
)

def query2vec(question):
    model_path = "bge-m3"
    #加载text2vec模型
    model = SentenceModel(model_path)
    embedding = model.encode(question).tolist() #获得嵌入,要list类型的
    return embedding

def search_top(question, top_k):
    query_vec = query2vec(question)  # 获取问题的嵌入向量

    start_time1 = time.time()
    
    # 执行Elasticsearch查询
    response = es.search(
        index=index_name,
        body={
            "size":top_k,
            "query": {
                "knn": {
                    "field": "embedding",
                    "query_vector": query_vec,
                    "k": top_k,
                }
            }
        }
    )

    end_time1 = time.time()  # 记录结束时间
    execution_time1 = end_time1 - start_time1  # 计算运行时间
    print(f"搜索时间为: {execution_time1} 秒")
    
    # 返回结果
    return response['hits']['hits']

if __name__ =="__main__":
    start_time2 = time.time()
    kk = search_top("如何通过人工智能推动地区的经济发展？",10)
    for i in kk:
        print(i['_source']['filename'])
    
    end_time2 = time.time()  # 记录结束时间
    execution_time2 = end_time2 - start_time2  # 计算运行时间
    print(f"程序总运行时间为: {execution_time2} 秒")
