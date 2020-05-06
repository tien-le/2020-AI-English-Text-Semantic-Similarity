# 2020-AI-English-Text-Semantic-Similarity
2020 AI研习社 英文文本语义相似度

## TODO
  
-[x]    句子切割成短语后, 生成多个text_a和text_b对, 这些text_a和text_b可能会混淆对应.
-[x]
        baseline:text_a和text_b, 训练和验证集随机划分:
        线上结果：88.62
        
-[x]        
        5折 + [(text_a, text_b), (text_b, title_a)] :
        -e=3 , -b=3
        线下结果：
        -e=3 , -b=24
        线上结果：89.9392        

-[x]
        5折 + [(text_a, text_b), (text_b, title_a)] + 英文全部转小写:
        -e=3 , -b=3
        线下结果：
        -e=3 , -b=24
        线上结果：90.6        

-[x]
        5折 + [(text_a, text_b), (text_b, title_a)] + 英文全部转小写 + 拼写检查:
        -e=3 , -b=3
        线下结果：
        -e=3 , -b=24
        线上结果：90.35
        
-[x]   
        5折 + [(text_a, text_b), (text_b, title_a)] + 英文全部转小写 + 拼写检查 + 伪标签:
        -e=3 , -b=3
        线下结果：
        -e=3 , -b=24
        线上结果：90.35

-[x] 

        Longformer