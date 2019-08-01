import pandas as pd
from helper import *
# ! pip install pandas nltk gensim pyldavis
# Ignore warnings
import warnings
warnings.filterwarnings('ignore')
from gensim.models.coherencemodel import CoherenceModel
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords 
from nltk.stem.wordnet import WordNetLemmatizer
import string

stopwords = set(stopwords.words('english'))
punctuation = set(string.punctuation) 
lemmatize = WordNetLemmatizer()

def cleaning(article):
    one = " ".join([i for i in article.lower().split() if i not in stopwords])
    two = "".join(i for i in one if i not in punctuation)
    three = " ".join(lemmatize.lemmatize(i) for i in two.split())
    return three

df = pd.read_csv(open("abstracts for 'automat'.csv", errors='ignore'))
df=df.astype(str)

text = df.applymap(cleaning)['paperAbstract']
text_list = [i.split() for i in text]

#add log for recording the model fitting data while training

from time import time
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO,
                   filename='running.log',filemode='w')


#build dictionary

# Importing Gensim
import gensim
from gensim import corpora

# Creating the term dictionary of our courpus, where every unique term is assigned an index. dictionary = corpora.Dictionary(doc_clean)
dictionary = corpora.Dictionary(text_list)
dictionary.save('dictionary.dict')


#build corpus

# Converting list of documents (corpus) into Document Term Matrix using dictionary prepared above.
doc_term_matrix = [dictionary.doc2bow(doc) for doc in text_list]
corpora.MmCorpus.serialize('corpus.mm', doc_term_matrix)


#Running LDA Model

start = time()
# Creating the object for LDA model using gensim library
Lda = gensim.models.ldamodel.LdaModel

# Running and Training LDA model on the document term matrix.
ldamodel = Lda(doc_term_matrix, num_topics=10, id2word = dictionary, passes=1, iterations=3)
#print 'used: {:.2f}s'.format(time()-start)

#print(ldamodel.print_topics(num_topics=2, num_words=4))

for i in ldamodel.print_topics():
    for j in i:
        print (j) ###########################
    

#save model for future use
    
ldamodel.save('topic.model')


#load saved model

from gensim.models import LdaModel
loading = LdaModel.load('topic.model')

#print(loading.print_topics(num_topics=2, num_words=4))


#PLOTTING

import pyLDAvis.gensim
import gensim
pyLDAvis.enable_notebook()

d = gensim.corpora.Dictionary.load('dictionary.dict')
c = gensim.corpora.MmCorpus('corpus.mm')
lda = gensim.models.LdaModel.load('topic.model')

data = pyLDAvis.gensim.prepare(lda, c, d)

text_file = open('LDA.txt', 'w')

text_file.write(str(data))

text_file.close()
 
pyLDAvis.save_html(data,'LDA Visualisation.html')


#Plot words importance

import gensim
import warnings
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
lda = gensim.models.LdaModel.load('topic.model')

fiz=plt.figure(figsize=(15,30))
for i in range(10):
    df=pd.DataFrame(lda.show_topic(i), columns=['term','prob']).set_index('term')
    df=df.sort_values('prob')
    
    plt.subplot(5,2,i+1)
    plt.title('topic '+str(i+1))
    sns.barplot(x='prob', y=df.index, data=df, label='Cities', palette='Reds_d')
    plt.xlabel('probability')

plt.show()

for index, topic in lda.show_topics(formatted=False, num_words= 30):
    print ('Topic {} \nWords: {}'.format(index+1, [w[0] for w in topic]))
    top_words=[]
    for w in topic:
        top_words.append(w[0]) 
#    print (top_words) #list view of top_words
#    print ('\n')