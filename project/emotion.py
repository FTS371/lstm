# 导入NLTK库和情感词典
import nltk
from nltk.corpus import sentiwordnet as swn

# 对文本进行分词
text = "This stock is going to the moon!"
tokens = nltk.word_tokenize(text)

# 对每个单词进行情感分析
sentiment = 0
for token in tokens:
    synsets = list(swn.senti_synsets(token))
    if synsets:
        # 取第一个情感词典的极性得分
        sentiment += synsets[0].pos_score() - synsets[0].neg_score()

# 输出情感极性得分
if sentiment > 0:
    print("Positive sentiment!")
elif sentiment < 0:
    print("Negative sentiment!")
else:
    print("Neutral sentiment.")
