#随机生成es中的内容，并一条一条写入，速度较慢，适合小规模数据
from elasticsearch_dsl import Document, Text, Index, connections,DenseVector,Integer
import numpy as np
import warnings
from tqdm import tqdm

warnings.filterwarnings('ignore')

#首先建立连接
client = connections.create_connection(hosts=['https://10.1.48.56:9200'], timeout=20,http_auth=('elastic','elastic'),verify_certs=False)

#指定索引名
index_name = 'test100w'
index = Index(index_name)

class VecDocument(Document):#存储的格式与索引
    filename = Text()
    num = Integer()
    text = Text()
    embedding = DenseVector(dims = 1024)#模型输出的维度为768

    class Index:
        name = index_name

#创建索引并将映射与索引关联
VecDocument.init()

dims = 1024#设置生成向量的维度

for i in tqdm(range(1000000), desc="Processing documents"):
    knowledge_document = VecDocument(filename = str(i),
                                    num = str(i),
                                    text = str(i),
                                    vector =  np.random.random(dims).tolist()
                                    )
    knowledge_document.save(using=client)
