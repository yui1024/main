from sklearn.datasets import fetch_20newsgroups
import numpy as np
from sklearn.model_selection import train_test_split

import sklearn
import pandas as pd

from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer

newsgroups = fetch_20newsgroups(subset='all')

def extract_inf(text, key='From:'):
    # 検索対象の文字列.find(検索する文字列)ー＞見つかった文字列の開始位置を返す
    start = text.find(key)
    # .find()関数は見つからなかったら-1を返す関数になっている
    if start == -1:
        # 見つからなかったので返すものはなし
        return None
    text = text[start + len(key) + 1:]
    # テキストの行のうち文字列の開始位置+fromなどのkeyの文字数+1(:の分の文字数)
    end = text.find('\n')
    # 開業が見つかったら
    result = text[:end]
    # a[:6] #[0, 1, 2, 3, 4, 5]endまでの列番号をもつ単語をresultに入れる
    # 'From:'なものが付いている行を全て出力
    return result

def remove_header(text):
    """
    ヘッダーの部分を除去
    """
    raw = text
    #２回改行があるときの開始位置をstartとする
    start = text.find('\n\n')
    text = text[start + 2:]
    while True:
        #例外処理を行う
        try:
            if text[0] == "\n":
                #文字列をtextに指定できるようにする
                text = text[1:]
            else:
                #改行んもなければ、文字列なのでtextとして取得する
                return text
        except:
            return raw

        # try:
        #     例外が発生するかもしれないが、実行したい処理。
        # except エラー名:
        #     例外発生時に行う処理



dataset = sklearn.datasets.fetch_20newsgroups(subset='all')
data = pd.DataFrame({
    "raw": dataset['data'],
    "file": dataset['filenames'],
    "target": dataset['target']
})
t_names = dataset['target_names']
# extract information

#pandas　Seriesは一次元
#applyは、無名関数（lambda）やdefで定義した関数をapply()の引数に渡す
data['author'] = data['raw'].apply(lambda x: extract_inf(x, 'From:'))
#authorはdataのraw行にkeyがfromである行(fromと書かれた行)を渡す
data['subject'] = data['raw'].apply(lambda x: extract_inf(x, 'Subject:'))
#subjectはdataのraw行にkeyがsubjectである行を渡す
data['line'] = data['raw'].apply(lambda x: extract_inf(x, 'Lines:'))
#lineはlinesと書かれた行を渡す
data['Organization'] = data['raw'].apply(lambda x: extract_inf(x, 'Organization:'))
data['text'] = data['raw'].apply(lambda x: remove_header(x))
# remove headerで二回改行されていなく、文字列で取ったもののみ
data['target_name'] = data['target'].apply(lambda x: t_names[x])

#各文章の単語数を取得する
data['word_count'] = data['text'].apply(lambda x: len(str(x).split(" ")))
data[['text','word_count']].head()
#各単語の単語数を表示
word_count=data.word_count.describe()
print(word_count)

#よくある単語を取得する
freq = pd.Series(' '.join(data['text']).split()).value_counts()[:20]
#特徴的な単語を取得する
not_freq =  pd.Series(' '.join(data['text']).split()).value_counts()[-20:]




# Libraries for text preprocessing
import re
import nltk
#nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import RegexpTokenizer
#nltk.download('wordnet')
from nltk.stem.wordnet import WordNetLemmatizer

##Creating a list of stop words and adding custom stopwords
stop_words = set(stopwords.words("english"))
##Creating a list of custom stopwords
new_words = ["using", "show", "result", "large", "also", "iv", "one", "two", "new", "previously", "shown"]
stop_words = stop_words.union(new_words)

corpus = []
for i in range(0, 3847):
    #句読点を消去
    text = re.sub('[^a-zA-Z]', ' ', data['text'][i])

    #小文字にかえる
    text = text.lower()

    #タグを取り除く
    text = re.sub("&lt;/?.*?&gt;", " &lt;&gt; ", text)

    #特殊文字と数字を削除
    text = re.sub("(\\d|\\W)+", " ", text)

    ##文字列からリストに変換
    text = text.split()

    ##Stemming
    # ステミングは、接尾辞を削除してテキストを正規化
    ps = PorterStemmer()

    # Lemmatisation
    # 語彙化は、単語の意味に基づいて機能するより高度な手法。単語を文章に合わせて品詞を変える
    lem = WordNetLemmatizer()
    text = [lem.lemmatize(word) for word in text if not word in stop_words]
    text = " ".join(text)
    corpus.append(text)

#Word cloud
from os import path
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import matplotlib.pyplot as plt
#%matplotlib inline

"""頻繁に使用される単語の視覚化"""
# wordcloud = WordCloud(
#                           background_color='white',
#                           stopwords=stop_words,
#                           max_words=100,
#                           max_font_size=50,
#                           random_state=42
#                          ).generate(str(corpus))
# print(wordcloud)
# fig = plt.figure(1)
# plt.imshow(wordcloud)
# plt.axis('off')
# plt.show()
# fig.savefig("word1.png", dpi=900)
#
#コーパス内のテキストは、機械学習アルゴリズムで解釈できる形式に変換する必要がある

#トークン化とは、連続したテキストを単語のリストに変換するプロセス
#整数のマトリックスに変換
from sklearn.feature_extraction.text import CountVectorizer
import re
#CountVectoriserを使用してテキストをトークン化、リスト化
cv=CountVectorizer(max_df=0.8,stop_words=stop_words, max_features=10000, ngram_range=(1,3))
#CountVectoriserクラスの変数「cv」を作成
X=cv.fit_transform(corpus)
#fit_transform関数を呼び出して語彙を学習および構築

#max_dfは、語彙を構築するとき、指定された閾値（コーパス固有のストップワード）
# より厳密に高いドキュメント頻度を持つ用語を無視。
# これは、コンテキストに関連する単語のみを使用し、一般的に使用される単語は使用しないようにする
#max_features —マトリックスの列数を決定
#ngram_rangeは、単一の単語、2つの単語（バイグラム）、および3つの単語（トリグラム）の組み合わせのリストを確認
# max_dfは0.8（８割以上の文書に出現する言葉はいらん）を設定
#vec_max_featuresは、整数を指定した場合，その個数
#(1, 3)なら1~3-gram
#n-gramとは文章をn個の単語ごとに分解する方法
print(list(cv.vocabulary_.keys())[:10])

from sklearn.feature_extraction.text import TfidfTransformer
tfidf_transformer = TfidfTransformer(smooth_idf=True, use_idf=True)
tfidf_transformer.fit(X)

#機能名を取得
feature_names = cv.get_feature_names()

# キーワードを抽出する必要がある
doc = corpus[532]

# 指定されたドキュメントのtf-idfを生成
tf_idf_vector = tfidf_transformer.transform(cv.transform([doc]))
print(tf_idf_vector)

#TF-IDFスコアに基づいて、最高スコアの単語を抽出して、ドキュメントのキーワードを取得できる
#tf_idfを降順でソートするための関数
from scipy.sparse import coo_matrix
def sort_coo(coo_matrix):
    tuples = zip(coo_matrix.col, coo_matrix.data)
    return sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)

def extract_topn_from_vector(feature_names, sorted_items, topn=10):
    """上位n個のアイテムの機能名とtf-idfスコアを取得"""

    # ベクトルの上位n個のアイテムのみを使用する
    sorted_items = sorted_items[:topn]

    score_vals = []
    feature_vals = []

    # word index and corresponding tf-idf score
    for idx, score in sorted_items:
        # keep track of feature name and its corresponding score
        score_vals.append(round(score, 3))
        feature_vals.append(feature_names[idx])

    # 機能名とそれに対応するスコアを追跡する
    # results = zip(feature_vals,score_vals)
    results = {}
    for idx in range(len(feature_vals)):
        results[feature_vals[idx]] = score_vals[idx]

    return results


# スコアの降順でtf-idfベクトルをソートする
sorted_items = sort_coo(tf_idf_vector.tocoo())
# extract only the top n; n here is 10
keywords = extract_topn_from_vector(feature_names, sorted_items, 20)

# now print the results
print("\nAbstract:")
print(doc)
print("\nKeywords:")
for k in keywords:
    print(k, keywords[k])






