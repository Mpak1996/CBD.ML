#!/usr/bin/env python
# coding: utf-8

# # CyberBullying Detection using NLP & Machine Learning (Dataset creation)

# ### 1. Libraries

# In[372]:


import pandas as pd
import numpy as np
import seaborn as sns 
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import re
import string
import matplotlib.pyplot as plt
from wordcloud import WordCloud


# ### 2. Import Datasets

# In[373]:


df_cyber = pd.read_csv("Datasets/cyberbullying_tweets.csv")
df_twitter = pd.read_csv("Datasets/twitter_parsed_dataset.csv")
df_twitter_racism = pd.read_csv("Datasets/twitter_racism_parsed_dataset.csv")
df_twitter_sexism = pd.read_csv("Datasets/twitter_sexism_parsed_dataset.csv")
df_youtube = pd.read_csv("Datasets/youtube_parsed_dataset.csv")


# ### 3.  Check first 10 lines of any dataset

# #### 3.1  cyberbullying_tweets.csv

# In[374]:


df_cyber.head(10)


# #### 3.2  twitter_parsed_dataset.csv

# In[375]:


df_twitter.head(10)


# #### 3.3  twitter_racism_parsed_dataset.csv

# In[376]:


df_twitter_racism.head(10)


# #### 3.4  twitter_sexism_parsed_dataset.csv

# In[377]:


df_twitter_sexism.head(10)


# #### 3.5  youtube_parsed_dataset.csv

# In[378]:


df_youtube.head(10)


# ### 4. Check Datasets type values and sum

# #### 4.1  cyberbullying_tweets.csv

# In[379]:


df_cyber.info()


# In[380]:


df_cyber['cyberbullying_type'].value_counts()


# 

# #### 4.2  twitter_parsed_dataset.csv

# In[381]:


df_twitter.info()


# In[382]:


df_twitter['Annotation'].value_counts()


# #### 4.3  twitter_racism_parsed_dataset.csv

# In[383]:


df_twitter_racism.info()


# In[384]:


df_twitter_racism['Annotation'].value_counts()


# #### 4.4  twitter_sexism_parsed_dataset.csv

# In[385]:


df_twitter_sexism.info()


# In[386]:


df_twitter_sexism['Annotation'].value_counts()


# #### 4.5 youtube_parsed_dataset.csv

# In[387]:


df_youtube.info()


# In[388]:


df_youtube['oh_label'].value_counts()


# ### 5. Graph of any Dataset

# #### 5.1 cyberbullying_tweets.csv

# In[389]:


sns.set(rc={'figure.figsize':(10.7,7.7)})
sns.countplot(x='cyberbullying_type', data=df_cyber).set(title="Graph of cyberbullying_tweets.csv")


# 

# #### 5.2 twitter_parsed_dataset.csv

# In[390]:


sns.set(rc={'figure.figsize':(10.7,7.7)})
sns.countplot(x='Annotation', data=df_twitter).set(title="Graph of twitter_parsed_dataset.csv")


# #### 5.3 twitter_racism_parsed_dataset.csv

# In[391]:


sns.set(rc={'figure.figsize':(10.7,7.7)})
sns.countplot(x='Annotation', data=df_twitter_racism).set(title="Graph of twitter_racism_parsed_dataset.csv")


# #### 5.4 twitter_sexism_parsed_dataset.csv

# In[392]:


sns.set(rc={'figure.figsize': (10.7, 7.7)})
sns.countplot(x='Annotation', data=df_twitter_sexism).set(title="Graph of twitter_sexism_parsed_dataset.csv")


# #### 5.5 youtube_parsed_dataset.csv

# In[393]:


sns.set(rc={'figure.figsize':(10.7,7.7)})
sns.countplot(x='oh_label', data=df_youtube).set(title="Graph of youtube_parsed_dataset.csv")


# ### 6. Classification for cyberbullying/not cyberbullying & rename columns

# #### 6.1 cyberbullying_tweets.csv

# In[394]:


df_cyber["cyberbullying_type"] = df_cyber["cyberbullying_type"].replace({"not_cyberbullying": 0})
df_cyber["cyberbullying_type"] = df_cyber["cyberbullying_type"].replace({"ethnicity": 1})
df_cyber["cyberbullying_type"] = df_cyber["cyberbullying_type"].replace({"age": 1})
df_cyber["cyberbullying_type"] = df_cyber["cyberbullying_type"].replace({"gender": 1})
df_cyber["cyberbullying_type"] = df_cyber["cyberbullying_type"].replace({"religion": 1})
df_cyber["cyberbullying_type"] = df_cyber["cyberbullying_type"].replace({"other_cyberbullying": 1})
df_cyber.to_csv('ExportedDatasets/cyberbullying_tweets/cyberbullying_tweets_class.csv', index=False)


# In[395]:


df_cyber.head()


# In[396]:


df_cyber.tail()


# In[397]:


df_cyber.info()


# #### 6.2 twitter_parsed_dataset.csv
# 

# In[398]:


df_twitter = df_twitter.rename(columns=({'Text':'tweet_text'}))
df_twitter = df_twitter.rename(columns=({'oh_label':'cyberbullying_type'}))
df_twitter = df_twitter.dropna()
df_twitter['cyberbullying_type'] = df_twitter['cyberbullying_type'].astype(int)
df_twitter.info()


# #### 6.3 twitter_racism_parsed_dataset.csv

# In[399]:


df_twitter_racism = df_twitter_racism.rename(columns=({'Text':'tweet_text'}))
df_twitter_racism = df_twitter_racism.rename(columns=({'oh_label':'cyberbullying_type'}))
df_twitter_racism = df_twitter_racism.dropna()
df_twitter_racism['cyberbullying_type'] = df_twitter_racism['cyberbullying_type'].astype(int)
df_twitter_racism.info()


# #### 6.4 twitter_sexism_parsed_dataset.csv

# In[400]:


df_twitter_sexism = df_twitter_sexism.rename(columns=({'Text':'tweet_text'}))
df_twitter_sexism = df_twitter_sexism.rename(columns=({'oh_label':'cyberbullying_type'}))
df_twitter_sexism = df_twitter_sexism.dropna()
df_twitter_sexism['cyberbullying_type'] = df_twitter_sexism['cyberbullying_type'].astype(int)
df_twitter_sexism.info()


# #### 6.5 youtube_parsed_dataset.csv

# In[401]:


df_youtube = df_youtube.rename(columns=({'Text':'tweet_text'}))
df_youtube = df_youtube.rename(columns=({'oh_label':'cyberbullying_type'}))
df_youtube = df_youtube.dropna()
df_youtube['cyberbullying_type'] = df_youtube['cyberbullying_type'].astype(int)
df_youtube.info()


# ### 7. Cleaning the tweet text

# In[402]:


def clean_tweets(tweet):
    # remove URL
    tweet = re.sub(r'http\S+', '', tweet)
    # Remove usernames
    tweet = re.sub(r'@[^\s]+[\s]?','',tweet)
    # Remove hastags
    tweet = re.sub(r'#[^\s]+[\s]?','',tweet)
    # remove special characters 
    tweet = re.sub('[^ a-zA-Z0-9]' , '', tweet)
    # remove RT
    tweet = re.sub('[RT]' , '', tweet)
    # remove Numbers
    tweet = re.sub('[0-9]', '', tweet)
    
    return tweet


# #### 7.1 cyberbullying_tweets.csv

# In[403]:


df_cyber["tweet_text"] = df_cyber["tweet_text"].apply(clean_tweets)
df_cyber.to_csv('ExportedDatasets/cyberbullying_tweets/cyberbullying_tweets_remove.csv', index=False)
df_cyber.head(30)


# #### 7.2 twitter_parsed_dataset.csv
# 

# In[404]:


df_twitter["tweet_text"] = df_twitter["tweet_text"].apply(clean_tweets)
df_twitter.to_csv('ExportedDatasets/twitter_parsed_dataset/twitter_parsed_dataset_remove.csv', index=False)
df_twitter.head(30)


# #### 7.3 twitter_racism_parsed_dataset.csv
# 

# In[405]:


df_twitter_racism["tweet_text"] = df_twitter_racism["tweet_text"].apply(clean_tweets)
df_twitter_racism.to_csv('ExportedDatasets/twitter_racism_parsed_dataset/twitter_racism_parsed_dataset_remove.csv', index=False)
df_twitter_racism.head(30)


# #### 7.4 twitter_sexism_parsed_dataset.csv
# 

# In[406]:


df_twitter_sexism["tweet_text"] = df_twitter_sexism["tweet_text"].apply(clean_tweets)
df_twitter_sexism.to_csv('ExportedDatasets/twitter_sexism_parsed_dataset/twitter_sexism_parsed_dataset_remove.csv', index=False)
df_twitter_sexism.head(30)


# #### 7.5 youtube_parsed_dataset.csv

# In[407]:


df_youtube["tweet_text"] = df_youtube["tweet_text"].apply(clean_tweets)
df_youtube.to_csv('ExportedDatasets/youtube_parsed_dataset/youtube_parsed_dataset_remove.csv', index=False)
df_youtube.head(30)


# ### 8. Drop empty tweets & Drop unused columns

# #### 8.1 cyberbullying_tweets.csv

# In[408]:


df_cyber = df_cyber.astype({'cyberbullying_type': 'int32'})
df_cyber= df_cyber[df_cyber['tweet_text'] != ""]
df_cyber.to_csv('ExportedDatasets/cyberbullying_tweets/cyberbullying_tweets_remove_empty.csv', index=False)
df_cyber.head(30)


# In[409]:


df_cyber.info()


# In[410]:


df_cyber['cyberbullying_type'].value_counts()


# #### 8.2 twitter_parsed_dataset.csv

# In[411]:


df_twitter= df_twitter[df_twitter['tweet_text'] != ""]
df_twitter.to_csv('ExportedDatasets/twitter_parsed_dataset/twitter_parsed_dataset_remove_empty.csv', index=False)
df_twitter = df_twitter.drop(['index','id','Annotation'], axis=1)
df_twitter.head(30)


# In[412]:


df_twitter.info()


# In[413]:


df_twitter['cyberbullying_type'].value_counts()


# #### 8.3 twitter_racism_parsed_dataset.csv

# In[414]:


df_twitter_racism= df_twitter_racism[df_twitter_racism['tweet_text'] != ""]
df_twitter_racism.to_csv('ExportedDatasets/twitter_racism_parsed_dataset/twitter_racism_parsed_dataset_remove_empty.csv', index=False)
df_twitter_racism = df_twitter_racism.drop(['index','id','Annotation'], axis=1)
df_twitter_racism.head(30)


# In[415]:


df_twitter_racism.info()


# In[416]:


df_twitter_racism['cyberbullying_type'].value_counts()


# #### 8.4 twitter_sexism_parsed_dataset.csv

# In[417]:


df_twitter_sexism= df_twitter_sexism[df_twitter_sexism['tweet_text'] != ""]
df_twitter_sexism.to_csv('ExportedDatasets/twitter_sexism_parsed_dataset/twitter_sexism_parsed_dataset_remove_empty.csv', index=False)
df_twitter_sexism = df_twitter_sexism.drop(['index','id','Annotation'], axis=1)
df_twitter_sexism.head(30)


# In[418]:


df_twitter_sexism.info()


# In[419]:


df_twitter_sexism['cyberbullying_type'].value_counts()


# #### 8.5 youtube_parsed_dataset.csv

# In[420]:


df_youtube= df_youtube[df_youtube['tweet_text'] != ""]
df_youtube.to_csv('ExportedDatasets/youtube_parsed_dataset/youtube_parsed_dataset_remove_empty.csv', index=False)
df_youtube = df_youtube.drop(['index','UserIndex','Number of Comments', 'Number of Subscribers', 'Membership Duration', 'Number of Uploads', 'Profanity in UserID', 'Age'], axis=1)
df_youtube.head(30)


# In[421]:


df_youtube.info()


# In[422]:


df_youtube['cyberbullying_type'].value_counts()


# ### 9. Graph of clean Datasets

# #### 9.1 cyberbullying_tweets.csv

# In[423]:


sns.set(rc={'figure.figsize':(10.7,7.7)})
sns.countplot(x='cyberbullying_type', data=df_cyber).set(title="Graph of clean cyberbullying_tweets.csv")


# #### 9.2 twitter_parsed_dataset.csv

# In[424]:


sns.set(rc={'figure.figsize':(10.7,7.7)})
sns.countplot(x='cyberbullying_type', data=df_twitter).set(title="Graph of clean twitter_parsed_dataset.csv")


# #### 9.3 twitter_racism_parsed_dataset.csv

# In[425]:


sns.set(rc={'figure.figsize':(10.7,7.7)})
sns.countplot(x='cyberbullying_type', data=df_twitter_racism).set(title="Graph of clean twitter_racism_parsed_dataset.csv")


# #### 9.4 twitter_sexism_parsed_dataset.csv

# In[426]:


sns.set(rc={'figure.figsize':(10.7,7.7)})
sns.countplot(x='cyberbullying_type', data=df_twitter_sexism).set(title="Graph of clean twitter_sexism_parsed_dataset.csv")


# #### 9.5 youtube_parsed_dataset.csv

# In[427]:


sns.set(rc={'figure.figsize':(10.7,7.7)})
sns.countplot(x='cyberbullying_type', data=df_youtube).set(title="Graph of clean youtube_parsed_dataset.csv")


# ### 10.  WordCloud of clean Dataset

# #### 10.1 cyberbullying_tweets.csv

# In[428]:


text = ''.join(df_cyber["tweet_text"].tolist())


# In[429]:


''.join(df_cyber["tweet_text"].tolist())


# In[430]:


wordcloud = WordCloud(width=1920, height=1080).generate(text)
fig = plt.figure(figsize=(10,10))
plt.imshow(wordcloud)
plt.axis('off')
plt.tight_layout(pad=0)
plt.show()


# #### 10.2 twitter_parsed_dataset.csv

# In[431]:


text = ''.join(df_twitter["tweet_text"].tolist())


# In[432]:


''.join(df_twitter["tweet_text"].tolist())


# In[433]:


wordcloud = WordCloud(width=1920, height=1080).generate(text)
fig = plt.figure(figsize=(10,10))
plt.imshow(wordcloud)
plt.axis('off')
plt.tight_layout(pad=0)
plt.show()


# #### 10.3 twitter_racism_parsed_dataset.csv
# 

# In[434]:


text = ''.join(df_twitter_racism["tweet_text"].tolist())


# In[435]:


''.join(df_twitter_racism["tweet_text"].tolist())


# In[436]:


wordcloud = WordCloud(width=1920, height=1080).generate(text)
fig = plt.figure(figsize=(10,10))
plt.imshow(wordcloud)
plt.axis('off')
plt.tight_layout(pad=0)
plt.show()


# #### 10.4 twitter_sexism_parsed_dataset.csv
# 

# In[437]:


text = ''.join(df_twitter_sexism["tweet_text"].tolist())


# In[438]:


''.join(df_twitter_sexism["tweet_text"].tolist())


# In[439]:


wordcloud = WordCloud(width=1920, height=1080).generate(text)
fig = plt.figure(figsize=(10,10))
plt.imshow(wordcloud)
plt.axis('off')
plt.tight_layout(pad=0)
plt.show()


# #### 10.5 youtube_parsed_dataset.csv

# In[440]:


text = ''.join(df_youtube["tweet_text"].tolist())


# In[441]:


''.join(df_youtube["tweet_text"].tolist())


# In[442]:


wordcloud = WordCloud(width=1920, height=1080).generate(text)
fig = plt.figure(figsize=(10,10))
plt.imshow(wordcloud)
plt.axis('off')
plt.tight_layout(pad=0)
plt.show()

