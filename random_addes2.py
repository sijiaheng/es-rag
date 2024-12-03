#随机生辰embedding数据写入es，批写入，速度较快，适合大规模数据
from elasticsearch import Elasticsearch, helpers
import numpy as np
from tqdm import tqdm  # 用于显示进度条
import warnings

warnings.filterwarnings('ignore')
 
# 创建 Elasticsearch 客户端
es = Elasticsearch(
    hosts=['https://localhost:9200'],  # 替换为您的 Elasticsearch 集群地址
    # 如果您的 Elasticsearch 集群需要身份验证，请取消以下两行的注释，并提供正确的用户名和密码
    http_auth=('elastic', 'elastic'),
    # 如果您的 Elasticsearch 集群使用自签名证书，请将 verify_certs 设置为 False
    verify_certs=False,
    timeout=60
)
 
# 指定索引名
index_name = 'test100w'
 
# 如果索引不存在，则创建它（这里省略了详细的映射定义）
if not es.indices.exists(index=index_name):
    es.indices.create(index=index_name)
 

num_documents = 1000000  # 要添加的文档数量
vector_dimensions = 1024  # 向量的维度
clu_size = 100000
clu_num = int(num_documents/clu_size)



for j in tqdm(range(clu_num),desc="正在完成批次："):#100w以上内存不够，需要分批写
    # 准备批量写入的数据
    actions = []
    for i in tqdm(range(j*clu_size,(j+1)*clu_size), desc="Preparing documents"):
        # 生成一个随机向量
        vector = np.random.random(vector_dimensions).tolist()
        # 创建文档数据
        doc = {
            '_index': index_name,
            '_id': str(i),  # 文档ID
            '_source': {
                'filename': f'file_{i}',  # 文件名（可以是任何字符串）
                'num': i,  # 一个整数字段
                'text': f'This is document number {i}',  # 文本字段
                'embedding': vector  # 向量字段
            }
        }
        # 将文档添加到操作列表中
        actions.append(doc)

    print("正在写入，请稍等。。。。。。。。。")
    #批量写入数据到 Elasticsearch,批量写入无法显示进度条
    try:
        helpers.bulk(es, actions)
        print("Documents have been successfully bulk indexed.")
        del actions
    except Exception as e:
        print(f"An error occurred during bulk indexing: {e}")
        del actions