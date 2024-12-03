#主要是为了进行向量搜索，返回匹配的top-n的知识
#采用余弦相似度搜索，精确knn搜索
import pandas as pd
import numpy as np
import transformers
from text2vec import SentenceModel
import torch.nn.functional as F
import numpy as np
from elasticsearch_dsl import Search, Document, Index, connections, DenseVector
from elasticsearch_dsl.query import ScriptScore, Q, MatchAll
import time
import warnings

warnings.filterwarnings('ignore')#忽略警告

#与es建立连接
client = connections.create_connection(hosts = ["https://localhost:9200"],timeout=60,http_auth=('elastic','elastic'),verify_certs=False)
# client = connections.create_connection(hosts = ['http://localhost:9200'],timeout = 20)
#指定索引
index_name = "test1e"#索引名字

def query2vec(question):
    model_path = "bge-m3"
    #加载text2vec模型
    model = SentenceModel(model_path)
    embedding = model.encode(question).tolist() #获得嵌入,要list类型的
    return embedding

def search_top(question,top_k):
    embedding = query2vec(question)#获得问题嵌入

    search = Search(index = index_name,using = client)
    start_time1 = time.time()
    # 计算余弦相似度
    script = {
    "source": "cosineSimilarity(params.query_vector, 'embedding') + 1.0",
    "params": {"query_vector": embedding}
    }
    query = MatchAll()
    script_score = ScriptScore(query=query,script=script)
    s = search.query(script_score)[0:top_k]

    response = s.execute()

    knowledge = [] #构建所需知识库

    for hit in response:
        if hit.meta.score >1.5:
            knowledge.append({"filename":hit.filename,"num":hit.num,"text":hit.text,"score":hit.meta.score})

    end_time1 = time.time()  # 记录结束时间
    execution_time1 = end_time1 - start_time1  # 计算运行时间
    print(f"搜索时间为: {execution_time1} 秒")
    return knowledge

if __name__ =="__main__":
    start_time2 = time.time()
    kk = search_top("如何通过人工智能推动地区的经济发展？",10)
    for i in kk:
        print(i)
    
    end_time2 = time.time()  # 记录结束时间
    execution_time2 = end_time2 - start_time2  # 计算运行时间
    print(f"程序总运行时间为: {execution_time2} 秒")



