#删除索引代码
from elasticsearch_dsl import Document, Text, Index, connections,DenseVector,Integer
from elasticsearch_dsl.query import MoreLikeThis

client = connections.create_connection(hosts=['https://10.1.48.56:9200'], timeout=20,http_auth=('elastic','elastic'),verify_certs=False)
# client = connections.create_connection(hosts = ['https://localhost:9200'],timeout = 20,http_auth=('elastic','elastic'),verify_certs=False)
#指定索引
index_name = "test100w"#索引名字
index = Index(index_name)
if index.exists():
    index.delete()