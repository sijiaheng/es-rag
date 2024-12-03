import gradio as gr
import pandas as pd
import numpy as np
import transformers
from text2vec import SentenceModel
import torch.nn.functional as F
import numpy as np
from elasticsearch_dsl import Search, Document, Index, connections, DenseVector
from elasticsearch_dsl.query import ScriptScore, Q, MatchAll
import time

#与es建立连接
client = connections.create_connection(hosts=['https://10.1.48.56:9200'], timeout=20,http_auth=('elastic','elastic'),verify_certs=False)
#指定索引
index_name = "knowledge_test"#索引名字

def query2vec(question):
    model_path = "text2vec-base-chinese"
    #加载text2vec模型
    model = SentenceModel(model_path)
    embedding = model.encode(question).tolist() #获得嵌入,要list类型的
    return embedding

def search_top(question,top_k):
    embedding = query2vec(question)#获得问题嵌入

    search = Search(index = index_name,using = client)
    # start_time1 = time.time()
    # 计算余弦相似度
    script = {
    "source": "cosineSimilarity(params.query_vector, 'vector') + 1.0",
    "params": {"query_vector": embedding}
    }
    query = MatchAll()
    script_score = ScriptScore(query=query,script=script)
    s = search.query(script_score)[0:top_k]

    response = s.execute()

    knowledge = [] #构建所需知识库

    for hit in response:
        if hit.meta.score >1.2:
            knowledge.append({"filename":hit.filename,"num":hit.num,"text":hit.text,"score":hit.meta.score})

    # end_time1 = time.time()  # 记录结束时间
    # execution_time1 = end_time1 - start_time1  # 计算运行时间
    # print(f"程序运行时间1为: {execution_time1} 秒")
    return knowledge

# 搜索函数
def search(query):
    # 可以在这里集成一个搜索引擎或数据库查询
    # results = ["Result 1 for: " + query, "Result 2 for: " + query]
    results = []
    top_k = search_top(query,5)
    for item in top_k:
        results.append(item['filename'])
    results = list(set(results))
    results_str = "\n".join(results)
    return results_str

def rag_output(question):
    # 这里应该是调用RAG模型的代码，返回生成的文本
    # 以下是一个示例返回值
    return "RAG模型的输出示例：这是对问题的回答。"

# 使用 Gradio Blocks 创建 UI
with gr.Blocks() as demo:
    with gr.Column():
        gr.Markdown("## Search Interface")
        query = gr.Textbox(label="Enter your query")
        submit = gr.Button("Search")
        results = gr.Textbox(label="References", placeholder="Search results will appear here...")
        rag_results = gr.Textbox(label="RAG Model Output", placeholder="RAG model output will appear here...")
        submit.click(fn=search, inputs=query, outputs=results)
        submit.click(fn=rag_output, inputs=query, outputs=rag_results)

# 启动应用
demo.launch(share=True)