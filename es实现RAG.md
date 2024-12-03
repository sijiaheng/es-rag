# es实现RAG

## es配置

在进行所有的工作之前，需要保证es环境已经配置成功。

在本次的工作中，采取的是在本机上搭建es，而后使用python操作它。

es的搭建参考的工作是：

https://blog.csdn.net/weixin_43926608/article/details/134163201

但是不能照搬，否则可能会出错。

----------------------------

第一步，首先重新注册一个用户，因为es不支持sudo用户进行操作。

```sh
sudo adduser es   //es可以是任何值
//建立文件夹，更改权限，并将其归属归为es
udo chown -R es:es ./elastic //切换到elasticsearch-8.10.4目录同级
```

第二步，切换到es用户

而后下载elasticsearch8.10.4

```sh
wget https://artifacts.elastic.co/downloads/elasticsearch/elasticsearch-8.10.4-linux-x86_64.tar.gz
```

后续基本与文档中相同。

## Kibana配置

![img_v3_02g5_792db646-79b8-4e8b-bfd6-38723179927g](/home/PJLAB/sijiaheng/.config/LarkShell/sdk_storage/f8b32a112618729d8f0d832858a9931b/resources/images/img_v3_02g5_792db646-79b8-4e8b-bfd6-38723179927g.jpg)

需要注意的是，要在官网上下载相同版本的kibana，而后修改kibana的配置文件，kibana.yml，如上所示。

----------------------

两者的启动均是在其相对应的bin目录下，执行同名文件即可。

启动完毕即可通过对应端口9200-es，5601-kibana。

# es实现RAG

主要的代码包括以下五个：

![image-20241106152232697](/home/PJLAB/sijiaheng/.config/Typora/typora-user-images/image-20241106152232697.png)

另外需要注意，还有一个pdf文件保存路径，这里指定的路径为

```python
folder_path = "/data/my_Rag/knowledge"
```

### delete.py

这是一个删除索引的代码，很简单，代码里是建立与es的连接后，删除索引名为knowledge_test的索引。

### know2vec

这是一个将pdf文件，进行分段转为embedding的代码。

首先建立与es的链接，并检查是否有索引，如果有则删除，防止索引中有原来残留的document。

而后建立索引的格式，包括文件名，属于文件中的第几部分，内容是什么，以及embedding，embedding的维度为768。

而后是一个函数，目的是将文件夹中的pdf逐一读取，并且分段，分段的方法是检测到有一个。.？！后接了一个/n，则判断为一个分段，进行切分。

切分之后将内容转为embedding，**需要注意，es的embedding接受的是list，numpy类型的没办法，所以需要加一个tolist()**

```python
def extract_text_from_pdfs(folder_path):
    data = []
    # 定义分段标志的正则表达式，因模型很难识别超过256个字的，因此需要分段处理，超过将会被截断
    segment_pattern = r'[。\.\?!]\n'  # 匹配以句号、问号或感叹号结尾的段落
    #设定embedding模型
    model_path = "text2vec-base-chinese"
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
```

而后是建立连接，将上一个函数中返回的数据，逐一读取存入es中

```python
def upload_es(folder_path):
    # 建立连接
    data = extract_text_from_pdfs(folder_path)
    for row in data:
        knowledge_document = VecDocument(filename = row['filename'],
                                         num = row['num'],
                                         text = row['text'],
                                         vector = row['embedding']
                                         )
        knowledge_document.save(using=client)
```

### know_search

这是对es中的内容，按照问题转成的embedding进行搜索。

**需要注意，在命名变量时有个问题，原问题（没有转embedding的）和作为问题去问es的，不好都命名做query，因此前者命名为question，后者为query。**

依然是先建立链接，指定索引。

第一个函数：将问题转为embedding：

```python
def query2vec(question):
    model_path = "text2vec-base-chinese"
    #加载text2vec模型
    model = SentenceModel(model_path)
    embedding = model.encode(question).tolist() #获得嵌入,要list类型的
    return embedding
```

第二个函数就很简单，通过embedding，计算余弦相似度，而后返回最大的余弦相似度的document：

```python
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
        if hit.meta.score >1.5:
            knowledge.append({"filename":hit.filename,"num":hit.num,"text":hit.text,"score":hit.meta.score})

    # end_time1 = time.time()  # 记录结束时间
    # execution_time1 = end_time1 - start_time1  # 计算运行时间
    # print(f"程序运行时间1为: {execution_time1} 秒")
    return knowledge
```

### rag_answer

这是根据rag搜索到的知识，构建prompt，调用大模型api的代码，代码很简单，这里不做详细描述，用时只需要设定好key即可。

### random_addes

这是自动生成内容填充es的代码，用作测试需求。

思路也很简单，建立连接后，指定索引，并设定索引的格式，在这里不要进行下面的步骤：

```python
if index.exists():
    index.delete()
```

否则会删去原来的，pdf转换的document。

而后根据格式自动填充内容，这里慢的地方在于es的写，因为设置的是逐条写入。











