#python.exe -m pip install nltk pandas sklearn
#python.exe ntlkpunktdownload.py

#   Dataset used http://ai.stanford.edu/~amaas/data/sentiment/
#   
#   
#   

#using NLTK library, we can do lot of text preprocesing
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from  sklearn.metrics  import accuracy_score
import pandas as pd
import os
#function to split text into word
#tokens = word_tokenize("The quick brown fox jumps over the lazy dog")
#nltk.download('punkt')
#nltk.download('stopwords')
#stop_words = set(stopwords.words('english'))
#tokens = [w for w in tokens if not w in stop_words]
#print(tokens)
#porter = PorterStemmer()
#stems = []
#for t in tokens:    
#    stems.append(porter.stem(t))
#print(stems)

folder = 'Imdb'
labels = {'pos': 1, 'neg': 0}
df = pd.DataFrame()
for f in ('test', 'train'):    
    for l in ('pos', 'neg'):
        path = os.path.join(folder, f, l)
        for file in os.listdir (path) :
            with open(os.path.join(path, file),'r', encoding='utf-8') as infile:
                txt = infile.read()
            df = df.append([[txt, labels[l]]],ignore_index=True)
df.columns = ['review', 'sentiment']

reviews = df.review.str.cat(sep=' ')
#function to split text into word
tokens = word_tokenize(reviews)
vocabulary = set(tokens)
print(len(vocabulary))
frequency_dist = nltk.FreqDist(tokens)
sorted(frequency_dist,key=frequency_dist.__getitem__, reverse=True)[0:50]

#generates csv file for later use
df.to_csv('movie_data.csv', index=False, encoding='utf-8')

#Building a Classifier
#Classifier to identify sentiment of each movie review.
#From the IMDb dataset, divide test and training sets of 25000 each
X_train = df.loc[:24999, 'review'].values
y_train = df.loc[:24999, 'sentiment'].values
X_test = df.loc[25000:, 'review'].values
y_test = df.loc[25000:, 'sentiment'].values

#coverting the text corpus into the feature vectors
vectorizer = TfidfVectorizer()
train_vectors = vectorizer.fit_transform(X_train)
test_vectors = vectorizer.transform(X_test)
print(train_vectors.shape, test_vectors.shape)

#train the model on the training set using Naive Bayes Classifier
clf = MultinomialNB().fit(train_vectors, y_train)

#testing the performance of our model 
predicted = clf.predict(test_vectors)
print(accuracy_score(y_test,predicted))