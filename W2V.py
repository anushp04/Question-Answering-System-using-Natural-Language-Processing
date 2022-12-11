#!/usr/bin/env python
# coding: utf-8

# ### Import Libraries

# In[ ]:


get_ipython().system('pip3 install --upgrade gensim --quiet')
get_ipython().system('pip3 install fse --quiet')
get_ipython().system('pip3 install swifter --quiet')
get_ipython().system('pip3 install pickle5 --quiet')

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import json # to read json
import os
from tqdm import tqdm
import pandas as pd
import numpy as np
import itertools
# import swifter
from scipy.spatial.distance import cosine
from collections import Counter
from nltk.tokenize import sent_tokenize
from sklearn.metrics.pairwise import cosine_similarity
from gensim.utils import simple_preprocess
from gensim.models.word2vec import Word2Vec
from gensim.models.fasttext import FastText
from gensim.models import KeyedVectors
from gensim.parsing.preprocessing import remove_stopwords, preprocess_string
import gensim.downloader
tqdm.pandas()
import nltk
nltk.download('punkt')
from IPython.display import HTML as html_print
from gensim.models.callbacks import CallbackAny2Vec
from sklearn import metrics
import swifter
import pickle5 as pickle

## **CONVERTING JSON TO CSV**

def new_json_to_dataframe(file):

    f = open ( file , "r") 
    data = json.loads(f.read())               #loading the json file.
    iid = []                                  
    tit = []                                  #Creating empty lists to store values.
    con = []
    Que = []
    Ans_st = []
    Txt = []
    
    for i in range(len(data['data'])):       #Root tag of the json file contains 'title' tag & 'paragraphs' list.
        
        title = data['data'][i]['title']
        for p in range(len(data['data'][i]['paragraphs'])):  # 'paragraphs' list contains 'context' tag & 'qas' list.
            
            context = data['data'][i]['paragraphs'][p]['context']
            for q in range(len(data['data'][i]['paragraphs'][p]['qas'])):  # 'qas' list contains 'question', 'Id' tag & 'answers' list.
                
                question = data['data'][i]['paragraphs'][p]['qas'][q]['question']
                Id = data['data'][i]['paragraphs'][p]['qas'][q]['id']
                for a in range(len(data['data'][i]['paragraphs'][p]['qas'][q]['answers'])): # 'answers' list contains 'ans_start', 'text' tags. 
                    
                    ans_start = data['data'][i]['paragraphs'][p]['qas'][q]['answers'][a]['answer_start']
                    text = data['data'][i]['paragraphs'][p]['qas'][q]['answers'][a]['text']
                    
                    tit.append(title)
                    con.append(context)
                    Que.append(question)                    # Appending values to lists
                    iid.append(Id)
                    Ans_st.append(ans_start)
                    Txt.append(text)

    print('Done')      # for indication perpose.
    new_df = pd.DataFrame(columns=['Id','title','context','question','ans_start','text']) # Creating empty DataFrame.
    new_df.Id = iid
    new_df.title = tit           #intializing list values to the DataFrame.
    new_df.context = con
    new_df.question = Que
    new_df.ans_start = Ans_st
    new_df.text = Txt
    print('Done')      # for indication perpose.
    final_df = new_df.drop_duplicates(keep='first')  # Dropping duplicate rows from the create Dataframe.
    return final_df

dev_data_from_json = new_json_to_dataframe('/content/drive/MyDrive/NLP Project CS6120/Code/Data/dev-v1.1.json')

train_data_from_json = new_json_to_dataframe('/content/drive/MyDrive/NLP Project CS6120/Code/Data/train-v1.1.json')

print(train_data_from_json.shape)

print(dev_data_from_json.shape)


## **BASELINE MODELLING**

### Load Dataset

#path_dir =  os.path.dirname(os.getcwd())
#train_df = pd.read_csv(os.path.join(path_dir,r'data/interim/train_data.csv'))
#train_df = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/CS6120 NLP/NLP/Data/train_data.csv')
#val_df = pd.read_csv(os.path.join(path_dir,r'data/interim/val_data.csv'))
#val_df = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/CS6120 NLP/NLP/Data/val_data.csv')
train_data_from_json = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/CS6120 NLP/NLP/Data/train_data_from_json_1.csv')
dev_data_from_json = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/CS6120 NLP/NLP/Data/dev_data_from_json_1.csv')
#dev_data.drop('Unnamed: 0',axis=1,inplace = True)
train_data_from_json.tail(5)

def cstr(s, color='black'):
    return "<text style=color:{}>{}</text>".format(color, s)

for i in range(1006,1009):
    print(f'\033[96m CONTEXT : {train_data_from_json["context"].tolist()[i]}')
    print(f'\033[91m QUESTION : {train_data_from_json["question"].tolist()[i]}')
    print(f'\033[92m ANSWER : {train_data_from_json["text"].tolist()[i]}')
    print('\033[90m ' + '-'*170)

### Get whole answer sentences

def get_answer_context(df):
    length_context = 0
    answer = ""

    for sentence in sent_tokenize(df.context):
        length_context += len(sentence) + 1
        if df.ans_start <= length_context:
            if len(sentence) >= len(str(df.text)):
                if answer == "":
                    return sentence
                else:
                    return answer + " " + sentence
            else:
                answer += sentence

train_data_from_json['answer_sentences'] = train_data_from_json.progress_apply(lambda row: get_answer_context(row),axis = 1)
dev_data_from_json['answer_sentences'] = dev_data_from_json.progress_apply(lambda row: get_answer_context(row),axis = 1)

train_data_from_json.head()

### Preprocess context

context_df = pd.DataFrame(train_data_from_json['context'].unique().tolist(),columns=['context'])
# Use the gensim.utils function simple_preprocess
context_df['processed'] = context_df['context'].progress_apply(lambda x: simple_preprocess(x))

question_df = pd.DataFrame(train_data_from_json['question'].unique().tolist(),columns=['question'])
# Use the gensim.utils function simple_preprocess
question_df['processed'] = question_df['question'].progress_apply(lambda x: simple_preprocess(x))


UNK = '<UNK>'

# init callback class
class callback(CallbackAny2Vec):
    """
    Callback to print loss after each epoch
    """
    def __init__(self):
        self.epoch = 0

    def on_epoch_end(self, model):
        loss = model.get_latest_training_loss()
        if self.epoch == 0:
            print('Loss after epoch {}: {}'.format(self.epoch, loss))
        else:
            print('Loss after epoch {}: {}'.format(self.epoch, loss- self.loss_previous_step))
        self.epoch += 1
        self.loss_previous_step = loss

def avg_sentence_vector(words, model, num_features):
    if isinstance(model,gensim.models.word2vec.Word2Vec):
        word_vec_model = model.wv
    else:
        word_vec_model = model
    index2word_set = word_vec_model.index_to_key 
    #function to average all words vectors in a given paragraph
    featureVec = np.zeros((num_features,), dtype="float32")
    nwords = 0

    for word in words:
        if word in index2word_set:
            nwords = nwords+1
            featureVec = np.add(featureVec, word_vec_model[word])

    if nwords>0:
        featureVec = np.divide(featureVec, nwords)
#     print(featureVec)
    return featureVec

def get_cosine_similarity(context,question,model,vector_size=300):
    if isinstance(model,gensim.models.word2vec.Word2Vec):
        vocab = model.wv.key_to_index
    else:
        vocab = model.key_to_index
#     print(context,question)
    context_sents = sent_tokenize(context)
#     print(context_sents)
    processed_context = [simple_preprocess(sent) for sent in context_sents]
    processed_context = [[word if word in vocab else UNK for word in processed_context_sent]                         for processed_context_sent in processed_context]
#     print(processed_context)
    processed_question = simple_preprocess(question)
    processed_question = [word if word in vocab else UNK for word in processed_question]
    
    context_vectors = [np.array(avg_sentence_vector(processed_context_sent,model,vector_size)).reshape(1,-1) for processed_context_sent in processed_context]
    question_vector  = np.array(avg_sentence_vector(processed_question,model,vector_size)).reshape(1,-1)
#     print(len(context_vectors[0]))
#     print(cosine_similarity(np.array(context_vectors[0]).reshape(1,-1),np.array(question_vector).reshape(1,-1)))
    
    cosine_sim_list = [cosine_similarity(context_sent_vector,question_vector) for context_sent_vector in context_vectors]
    
#     print(f"Cosine scores: {cosine_sim_list}")
    max_cosine_sim = max(cosine_sim_list)
    predicted_answer = context_sents[np.argmax(cosine_sim_list)]
    return max_cosine_sim, predicted_answer

temp_df = train_data_from_json.tail(100)

### Download word2vec model google

print(list(gensim.downloader.info()['models'].keys()))

google_model = gensim.downloader.load('word2vec-google-news-300')

sample_context = train_data_from_json['context'].tolist()[10]
sample_question = train_data_from_json['question'].tolist()[10]
print(f"\033[96m CONTEXT:{sample_context}")
print(f"\033[91m QUESTION: {sample_question}")
get_cosine_similarity(sample_context,sample_question,google_model)

google_model['world'].shape

google_model.most_similar(positive=['university'], topn = 3)

### Evaluvate results

def avg_sentence_vector(words, model, num_features):
    if isinstance(model,gensim.models.word2vec.Word2Vec):
        word_vec_model = model.wv
    else:
        word_vec_model = model
    index2word_set = word_vec_model.index_to_key 
    #function to average all words vectors in a given paragraph
    featureVec = np.zeros((num_features,), dtype="float32")
    nwords = 0

    for word in words:
        if word in index2word_set:
            nwords = nwords+1
            featureVec = np.add(featureVec, word_vec_model[word])

    if nwords>0:
        featureVec = np.divide(featureVec, nwords)
#     print(featureVec)
    return featureVec
def get_context_vector(context,model,vector_size=300):
    if isinstance(model,gensim.models.word2vec.Word2Vec):
        vocab = model.wv.key_to_index
    else:
        vocab = model.key_to_index
    context_sents = sent_tokenize(context)
    processed_context = [simple_preprocess(sent) for sent in context_sents]
    processed_context = [[word if word in vocab else UNK for word in processed_context_sent]                         for processed_context_sent in processed_context]
    context_vectors = [np.array(avg_sentence_vector(processed_context_sent,model,vector_size)).reshape(1,-1) for processed_context_sent in processed_context]
    
    return context_vectors
    
def get_cosine_similarity(context,context_vectors,question,model,vector_size=300):
    context_sents = sent_tokenize(context)
    
    if isinstance(model,gensim.models.word2vec.Word2Vec):
        vocab = model.wv.key_to_index
    else:
        vocab = model.key_to_index
        
    processed_question = simple_preprocess(question)
    processed_question = [word if word in vocab else UNK for word in processed_question]
    
    question_vector  = np.array(avg_sentence_vector(processed_question,model,vector_size)).reshape(1,-1)
    
    cosine_sim_list = [cosine_similarity(context_sent_vector,question_vector) for context_sent_vector in context_vectors]
    
    max_cosine_sim = max(cosine_sim_list)
    predicted_answer = context_sents[np.argmax(cosine_sim_list)]
    return max_cosine_sim, predicted_answer

temp_df['context_vec'] = temp_df['context'].swifter.progress_bar(enable=True, desc=None).apply(lambda x: get_context_vector(x,google_model))

temp_df[['cosine_sim','predicted_answer']] = temp_df[['context','context_vec','question']].swifter.progress_bar(enable=True, desc=None).apply(lambda x: get_cosine_similarity(x[0],x[1],x[2],google_model,300),axis=1,result_type="expand")
temp_df.head(2)

temp_df['correct_prediction'] = temp_df['answer_sentences'] == temp_df['predicted_answer']
print(temp_df['correct_prediction'].value_counts())
print(f"accuracy: {temp_df[temp_df['correct_prediction']].shape[0]/temp_df.shape[0]}")

# temp_df[['consine_sim','predicted_answer']] = temp_df[['context','question']]\
# .swifter.progress_bar(enable=True, desc=None)\
# .apply(lambda x: get_cosine_similarity(x[0],x[1],google_model,300),axis=1,result_type="expand")
# temp_df.head(2)

#### On val set 

dev_data_from_json['context_vec'] = dev_data_from_json['context'].swifter.progress_bar(enable=True, desc=None).apply(lambda x: get_context_vector(x,google_model))

dev_data_from_json[['cosine_sim','predicted_answer']] = dev_data_from_json[['context','context_vec','question']].swifter.progress_bar(enable=True, desc=None).apply(lambda x: get_cosine_similarity(x[0],x[1],x[2],google_model,300),axis=1,result_type="expand")
dev_data_from_json.head(2)

dev_data_from_json.to_pickle('/content/drive/MyDrive/Colab Notebooks/CS6120 NLP/NLP/Data/contextual_dev_v1.pkl')


with open('/content/drive/MyDrive/Colab Notebooks/CS6120 NLP/NLP/Data/contextual_dev_v1.pkl', "rb") as fh:
  dev_data = pickle.load(fh)

dev_data['correct_prediction'] = dev_data['answer_sentences'] == dev_data['predicted_answer']
dev_data['correct_prediction'].value_counts()

print(f"accuracy: {dev_data[dev_data['correct_prediction']].shape[0]/dev_data.shape[0]}")


# In[ ]:




