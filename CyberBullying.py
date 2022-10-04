#!/usr/bin/env python
# coding: utf-8

# # CyberBullying Detection using NLP & Machine Learning

# ### 1. Libraries

# In[58]:


import pandas as pd
import numpy as np
import seaborn as sns 
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import re
import string
import matplotlib.pyplot as plt
from wordcloud import WordCloud


# ### 2. Import Dataset

# In[59]:


imdf_cyber = pd.read_csv("Dataset/cyberbullying_tweets.csv")


# ### 3.  Check first 10 lines

# In[60]:


imdf_cyber.head(10)


# ### 4. Check datasets type values and sum

# In[82]:


imdf_cyber.info()


# In[62]:


imdf_cyber['cyberbullying_type'].value_counts()


# ### 5. Graph of Dataset

# In[63]:


sns.set(rc={'figure.figsize':(10.7,7.7)})
sns.countplot(x='cyberbullying_type',data=imdf_cyber).set(title="Graph of Dataset")


# ### 6.  WordCloud of Dataset

# In[64]:


text = ''.join(imdf_cyber["tweet_text"].tolist())


# In[65]:


''.join(imdf_cyber["tweet_text"].tolist())


# In[66]:


wordcloud = WordCloud(width=1920, height=1080).generate(text)
fig = plt.figure(figsize=(10,10))
plt.imshow(wordcloud)
plt.axis('off')
plt.tight_layout(pad=0)
plt.show()


# ### 7. Classification for cyberbullying/not cyberbullying to 1/0

# In[67]:


imdf_cyber["cyberbullying_type"] = imdf_cyber["cyberbullying_type"].replace({"not_cyberbullying": 0})
imdf_cyber["cyberbullying_type"] = imdf_cyber["cyberbullying_type"].replace({"ethnicity": 1})
imdf_cyber["cyberbullying_type"] = imdf_cyber["cyberbullying_type"].replace({"age": 1})
imdf_cyber["cyberbullying_type"] = imdf_cyber["cyberbullying_type"].replace({"gender": 1})
imdf_cyber["cyberbullying_type"] = imdf_cyber["cyberbullying_type"].replace({"religion": 1})
imdf_cyber["cyberbullying_type"] = imdf_cyber["cyberbullying_type"].replace({"other_cyberbullying": 1})
imdf_cyber.to_csv('Dataset/cyberbullying_tweets_class.csv', index=False)


# In[68]:


imdf_cyber.head()


# In[69]:


imdf_cyber.tail()


# ### 8. Cleaning the tweet text

# In[70]:


def clean_tweets(tweet):
    # remove URL
    tweet = re.sub(r'http\S+', '', tweet)
    # Remove usernames
    tweet = re.sub(r'@[^\s]+[\s]?','',tweet)
    # remove special characters 
    tweet = re.sub('[^ a-zA-Z0-9]' , '', tweet)
    # remove Numbers
    tweet = re.sub('[0-9]', '', tweet)
    
    return tweet


# In[71]:


imdf_cyber["tweet_text"] = imdf_cyber["tweet_text"].apply(clean_tweets)


# In[72]:


imdf_cyber.to_csv('Dataset/cyberbullying_tweets_remove.csv', index=False)


# In[73]:


imdf_cyber.head(30)


# ### 9. Drop empty tweets

# In[74]:


imdf_cyber= imdf_cyber[imdf_cyber['tweet_text'] != ""]


# In[75]:


imdf_cyber.to_csv('Dataset/cyberbullying_tweets_remove_empty.csv', index=False)


# In[76]:


imdf_cyber.head(30)


# In[77]:


imdf_cyber.info()


# In[78]:


imdf_cyber['cyberbullying_type'].value_counts()


# ### 10. Graph of clean Dataset

# In[79]:


sns.set(rc={'figure.figsize':(10.7,7.7)})
sns.countplot(x='cyberbullying_type',data=imdf_cyber).set(title="Graph of clean Dataset")


# ### 11.  WordCloud of clean Dataset

# In[80]:


text = ''.join(imdf_cyber["tweet_text"].tolist())


# In[81]:


wordcloud = WordCloud(width=1920, height=1080).generate(text)
fig = plt.figure(figsize=(10,10))
plt.imshow(wordcloud)
plt.axis('off')
plt.tight_layout(pad=0)
plt.show()

