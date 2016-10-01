import pandas
import nltk
import numpy
import codecs
import sys
import pickle
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer
from nltk.stem.lancaster import LancasterStemmer
from gensim import corpora, models, similarities
from os import listdir
from os.path import isfile, join

''' Dataset Model'''
'''read data'''
df = pandas.read_csv('D:\\training\\test\\profile\\Profile.csv')
fwrite = codecs.open('D:\\algebra function\\bigtext.txt','a')
text_line=[]

for i in range(0,1201): 
    f = codecs.open('D:\\training\\text\\'+df['userid'][i]+'.txt', encoding='latin-1')
    text = f.readline()
    text_line.append(text)

print "ready to remove the stopwords and stems!"
''' Dataset Model Ends'''


'''Data Preprocessing Model'''
'''remove stop words'''
english_stopwords = stopwords.words("english")
for i in range (0,len(text_line)):
    texttemp = text_line[i]
    text_line[i] = texttemp.lower()

tknzr = TweetTokenizer()
for i in range (0,len(text_line)):
    text_line[i] = tknzr.tokenize(text_line[i])

texts_filtered_stopwords = [[word for word in document if not word in english_stopwords] for document in text_line]
'''stemming'''
st = LancasterStemmer()
texts_stemmed = [[st.stem(word) for word in docment] for docment in texts_filtered_stopwords ]
all_stems = sum(texts_stemmed, [])
stems_once = set(stem for stem in set(all_stems) if all_stems.count(stem) == 1)
texts = [[stem for stem in text if stem not in stems_once] for text in texts_stemmed]

training_texts = []
validation_texts = []
'''split dataset into 2 parts: training dataset and validation dataset'''
for i in range(0, 1201):
    if(i<1000):
        training_texts.append(texts[i])
    else:
        validation_texts.append(texts[i])

'''form dicationary'''
dictionary = corpora.Dictionary(training_texts)

corpus = [dictionary.doc2bow(text) for text in training_texts]
'''tf-idf'''
tfidf = models.TfidfModel(corpus)
corpus_tfidf = tfidf[corpus]
'''Data Preprocessing Model Ends'''
'''lsi form new vector space, with 34 dimensions'''
lsi = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=34)
cc_lsi = lsi[corpus_tfidf]
for doc in cc_lsi:
    print doc
'''get similarity indexes'''
index = similarities.MatrixSimilarity(lsi[corpus])

'''test'''
'''
algebra_course = validation_texts[1]
real_gender = df['gender'][1000 + 1]
algebra_bow = dictionary.doc2bow(algebra_course)
algebra_lsi = lsi[algebra_bow]
sims = index[algebra_lsi] 
sort_sims = sorted(enumerate(sims), key=lambda item: -item[1])
for j in range (0,7):

        row = sort_sims[j][0]
        print row
'''
'''test ends'''

'''KNN prediction Model'''
'''accuracy calculation'''
right = 0.0
male_count = 0
female_count = 0
for i in range(0,200):
    algebra_course = validation_texts[i]
    real_gender = df['gender'][1000 + i]
    algebra_bow = dictionary.doc2bow(algebra_course)
    algebra_lsi = lsi[algebra_bow]
    '''get similarity score based validation_texts[i]'''
    sims = index[algebra_lsi] 
    '''sort similarity from high to low'''
    sort_sims = sorted(enumerate(sims), key=lambda item: -item[1])

    male_vote = 0
    female_vote = 0
    for j in range (0,7):

        row = sort_sims[j][0]

        gender = df['gender'][row]
        if(gender == 0):
            male_vote += 1
        else:
            female_vote += 1
    '''KNN prediction Model Ends'''
    '''Accurary Calculate Model'''
    if(male_vote > female_vote):
        male_count += 1
        if(real_gender == 0):
          right += 1
    else:
        female_count += 1
        if(real_gender == 1):
            right += 1
    '''Accurary Calculate Model Ends'''
print right / 200    
