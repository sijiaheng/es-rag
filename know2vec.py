#pdf直接embedding后写入es数据库中，无任何中间数据可生成存储
import os
import pdfplumber
import pandas as pd
import re
import transformers
from text2vec import SentenceModel
from elasticsearch_dsl import Document, Text, Index, connections,DenseVector,Integer
import warnings

warnings.filterwarnings('ignore')#忽略警告

#知识库pdf存放位置
folder_path = "/data/ES_RAG/knowledge"
#与es建立连接
client = connections.create_connection(hosts=['https://localhost:9200'], timeout=20,http_auth=('elastic','elastic'),verify_certs=False)

#指定索引
index_name = "test100w"#索引名字
index = Index(index_name)
if index.exists():
    index.delete()

class VecDocument(Document):#存储的格式与索引
    filename = Text()
    num = Integer()
    text = Text()
    embedding = DenseVector(dims = 1024)#模型输出的维度为768

    class Index:
        name = index_name
#创建索引并将映射与索引关联
VecDocument.init()


def extract_text_from_pdfs(folder_path):
    data = []
    # 定义分段标志的正则表达式，因模型很难识别超过256个字的，因此需要分段处理，超过将会被截断
    segment_pattern = r'[。\.\?!]\n'  # 匹配以句号、问号或感叹号结尾的段落
    #设定embedding模型
    model_path = "bge-m3"
    model = SentenceModel(model_path)


    # 遍历文件夹中的所有PDF文件
    # 分段落转换为向量，存储在list中
    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(folder_path, filename)
            with pdfplumber.open(pdf_path) as pdf:
                i =0# 标记段落
                for page in pdf.pages:
                    text = page.extract_text()
                    if text:
                        # 按定义的标志分段
                        paragraphs = re.split(segment_pattern, text)
                        for paragraph in paragraphs:
                            stripped_paragraph = paragraph.strip()#删除段落两端的空白字符
                            stripped_paragraph = re.sub(r'\s+', '', stripped_paragraph)#删除段落内的空白字符
                            if stripped_paragraph:  # 只保留非空段落
                                embedding = model.encode(stripped_paragraph).tolist()
                                data.append({"filename": filename,"num":i, "text": stripped_paragraph,"embedding":embedding})
                                i = i+1
    return data



def upload_es(folder_path):
    # 建立连接
    data = extract_text_from_pdfs(folder_path)
    for row in data:
        knowledge_document = VecDocument(filename = row['filename'],
                                         num = row['num'],
                                         text = row['text'],
                                         embedding = row['embedding']
                                         )
        knowledge_document.save(using=client)

upload_es(folder_path)
