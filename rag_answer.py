#调用百度文心一言测试
#文心一言生成有cecretkey
#python方法
#这里采用的搜索是精确knn搜索，如果要更改，需要import对应的包，并调用其方法
import requests  
import json
import know_search #调用，获得所需知识


def GetAccessToken(APIKey, SecretKey):  
    url = "https://aip.baidubce.com/oauth/2.0/token?grant_type=client_credentials&client_id={}&client_secret={}".format(APIKey, SecretKey)  
    payload = json.dumps("")  
    headers = {'Content-Type': 'application/json', 'Accept': 'application/json'}  
    response = requests.request("POST", url, headers=headers, data=payload)  
    return response.json().get("access_token")  
  
def GetBaiduAi(question, model_url, APIKey, SecretKey):  
    try:  
        access_token = GetAccessToken(APIKey, SecretKey)  
        url = "{}?access_token=".format(model_url) + access_token  
        payload = json.dumps({  
            "temperature": 0.95,  
            "top_p": 0.7,  
            "system": '你是一名智能百科助手，你要尽可能的回答详尽、科学严谨，你要逐步的生成回答，逻辑性要强',  
            "messages": [  
                {  
                    "role": "user",  
                    "content": question  
                }  
            ]  
        })  
        headers = {'Content-Type': 'application/json'}  
        response = requests.request("POST", url, headers=headers, data=payload)  
        content = response.text  
        content = json.loads(content)  
        resultContent = content["result"]  
        return resultContent  
    except Exception as e:  
        print(e)  
        return str(e)
    

############################
######rag思路
######首先将问题进行向量化，然后矩阵乘得到向量相似度，最后排序返回最大，根据标签得到knowledge
############################

# if __name__ =="__main__":
#     APIKey = ''  
#     SecretKey = ''  
#     model_url = 'https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/ernie-lite-8k' 
#     query = '如何通过人工智能推动地区的经济发展？'
#     top_k = 5

#     searched_knowledge = know_search.search_top(query,top_k)
#     knowledge = ""
#     for row in searched_knowledge:
#         knowledge = knowledge + row['text'] + '\n'


#     prompt_template = f"""Answer questions with the follow knowledge:

#     Knowledge:{knowledge}

#     Query:{query}
#     """

#     print(prompt_template)

#     result =GetBaiduAi(prompt_template,model_url,APIKey,SecretKey)

#     print(result)


###下面是不使用rag的
if __name__ =="__main__":
    APIKey = ''#请填写你的apikey等等  
    SecretKey = ''  
    model_url = 'https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/ernie-lite-8k' 
    query = '如何通过人工智能推动地区的经济发展？'

    prompt_template = f"""Answer questions:

    Query:{query}
    """

    print(prompt_template)

    result =GetBaiduAi(prompt_template,model_url,APIKey,SecretKey)

    print(result)