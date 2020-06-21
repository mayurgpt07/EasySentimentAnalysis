import pandas as pd 
import numpy as np
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import warnings
import time
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense, Dropout, LSTM, Embedding, Masking, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.sequence import pad_sequences
warnings.filterwarnings("ignore")


def reduce_lengthening(text):
    pattern = re.compile(r"(.)\1{2,}")
    return pattern.sub(r"\1\1", text)


def featureEngineer(sentiments):
	htmlSyntax = re.compile('<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});')
	sentiments['review'] = sentiments['review'].apply(lambda x: re.sub('\ |\!|\/|\;|\:|\=|\"|\(|\)|\:|\]|\[|\<|\>|\{|\}|\'|\?|\.|\,|\|',' ', str(x)))
	sentiments['review'] = sentiments['review'].apply(lambda x: re.sub('\d+',' ', str(x)))	
	sentiments['review'] = sentiments['review'].apply(lambda x: re.sub(htmlSyntax,' ', str(x)))	
	sentiments['review'] = sentiments['review'].apply(lambda x: re.sub('\s+',' ',str(x).strip().lower()))
	sentiments['review'] = sentiments['review'].apply(lambda x: word_tokenize(x))
	sentenceList = []
	sentenceListTest = []
	for i in sentiments['review']:
		wordList = ''	
		for k in i:
			if k not in stopwordsList:
				wordList = wordList + reduce_lengthening(lemmatizer.lemmatize(k)) + ' '
		sentenceList.append(wordList.strip().lower())
	#wordList.clear()

	sentiments['review'] = pd.Series(sentenceList)
	sentiments['review'] = sentiments['review'].apply(lambda x: re.sub('\s+', ' ', str(x).strip()))
	return sentiments


sentimentData = pd.read_csv('IMDB Dataset.csv', header = 0)

stopwordsList = set(list(set(stopwords.words('english')))+['br'])
# print(list(stopwordsList))
lemmatizer = WordNetLemmatizer()

t0 = time.time()
sentimentData = featureEngineer(sentimentData)
t1 = time.time()
print('Time Take is:', t1 - t0)
sentimentData['sentiment'] = sentimentData['sentiment'].map({'positive': 1, 'negative': 0})

dataset_train, dataset_test, train_data_label, test_data_label = train_test_split(sentimentData['review'], sentimentData['sentiment'], test_size=0.20)

tokens = Tokenizer(num_words = 20000)

train = pd.DataFrame(dataset_train)
test = pd.DataFrame(dataset_test)
train.columns = ['review']
test.columns = ['review']

tokens.fit_on_texts(train['review'])
sequences = tokens.texts_to_sequences(train['review'])

tokens.fit_on_texts(test['review'])
sequences_test = tokens.texts_to_sequences(test['review'])


X_train = pad_sequences(sequences, maxlen = 200, padding = 'post')
X_test = pad_sequences(sequences_test, maxlen = 200, padding = 'post')
# embeddings = Embedding(input_dim = 30000, output_dim = 10, mask_zero = True)
# embeddingsData = embeddings(padding)

#Number of rows, max width or max features or maxlen after padding, output_dim of embeddings
# print(embeddingsData.shape)
y_train = train_data_label.copy()
y_test = test_data_label.copy()

print(X_train.shape, X_test.shape)

model = Sequential([Embedding(20000, 64, mask_zero=True),
                    Bidirectional(LSTM(64, dropout=0.2)),
                    # Dense(64, activation='relu'),
                    Dense(1, activation = 'sigmoid')])

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

history = model.fit(X_train, y_train,
                    batch_size=50,
                    epochs=10, validation_data = (X_test, y_test))