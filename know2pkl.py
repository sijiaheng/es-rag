#pdf数据切片-》存储在pkl数据格式中，每个pdf一个pkl数据包
import os
import pdfplumber
import pandas as pd
import re
import transformers
from text2vec import SentenceModel
import pickle

#知识库pdf存放位置
folder_path = "/data/my_Rag/data/中学题库/中考数学题库PDF"
pkl_path = "/data/ES_RAG/pkl/中学数学题库"

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
                data = []#每次存完之后，data存储清空
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
                #将data写入对应得pkl文件中
                # 保存 data 到指定的 pkl 文件中
                output_path = os.path.join(pkl_path, f"{os.path.splitext(filename)[0]}.pkl")
                with open(output_path, 'wb') as f:
                    pickle.dump(data, f)
                print(f"Data saved to {output_path}")
    return 0

extract_text_from_pdfs(folder_path)
