# %%
import pandas as pd

# %%
#!nvidia-smi -l 1

# %%
import tensorflow as tf
import pandas as pd
import numpy as np
import ktrain
from ktrain import text
import tensorflow as tf
import seaborn as sns
from nltk.tokenize import word_tokenize
from tensorflow.keras.preprocessing.text import Tokenizer

# %%
tf.__version__

# %%
#df = pd.read_csv('/Users/isingh/Desktop/IMDB_Dataset_cleaned_new.csv')

df = pd.read_csv('/workspaces/NLP_IBDM_Analysis_backup/IMDB_Dataset_cleaned_new.csv.zip')


# %%
df.head()

# %%
# reviews = df['review'].values
# number of positive and negative reviews
df['sentiment'].value_counts()

# %%
TRAIN_SIZE = 40000
TEST_SIZE = 10000

data_train = df[:TRAIN_SIZE]
data_test = df[TRAIN_SIZE:].reset_index(drop=True)

# %%
data_train.head()

# %%
data_test.head()

# %%
print("Size of train dataset: ",data_train.shape)
print("Size of test dataset: ",data_test.shape)

# %%
(X_train, y_train), (X_test, y_test), preproc = text.texts_from_df(train_df=data_train,text_column = 'review',label_columns = 'sentiment',val_df = data_test,maxlen = 500,ngram_range=2,preprocess_mode = 'bert')

# %%
len(X_train[1])

# %%
X_train[0].shape

# %%
model = text.text_classifier(name = 'bert',train_data = (X_train, y_train),preproc = preproc)

# %%
# fit the mode
learner = ktrain.get_learner(model = model,train_data = (X_train, y_train),val_data = (X_test, y_test),batch_size = 6)


# %%
learner.fit_onecycle(lr = 0.01, epochs = 1)

# %%
import transformers
# tokenizer import 
#from transformers import BertTokenizer


# %%
transformer = transformers.TFDistilBertModel.from_pretrained('distilbert-base-uncased')

# %%
MAX_VOCAB_SIZE = 10000

# %%
tokenizer = Tokenizer(num_words = MAX_VOCAB_SIZE,oov_token="<oov>")

# %%
# create a sequence of reviews
reviews = df['review'].values


# %%
seq_train = tokenizer.texts_to_sequences([X_train])
seq_test =  tokenizer.texts_to_sequences([X_test])
/-d614531/3
# %%
# fit the model
learner.fit_onecycle(lr = 0.01, epochs = 1)

# %%
# predictor = ktrain.get_predictor(learner.model, preproc)
# predictor.save("/Users/isingh/Desktop/IMDB_Dataset_cleaned_new.csv")


