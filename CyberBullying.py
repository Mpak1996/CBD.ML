#!/usr/bin/env python
# coding: utf-8

# # Cyber Bulling Detection using NLP & Machine Learning

# ### 1. Libraries

# In[69]:


import pandas as pd
import numpy as np
import seaborn as sns 
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import re
import string


# ### 2. Import Dataset

# In[70]:


imdf_cyber = pd.read_csv("Dataset/cyberbullying_tweets.csv")


# ### 3.  Check first 10 lines

# In[71]:


imdf_cyber.head(10)


# ### 4. Check datasets type values and sum

# In[72]:


imdf_cyber.info()


# ### 5. Graph of Dataset

# In[73]:


sns.set(rc={'figure.figsize':(10.7,7.7)})
sns.countplot(x='cyberbullying_type',data=imdf_cyber).set(title="Graph of Dataset")


# ### 6. Classification for cyberbulling/not cyberbulling to 1/0

# In[74]:


imdf_cyber["cyberbullying_type"] = imdf_cyber["cyberbullying_type"].replace({"not_cyberbullying": 0})
imdf_cyber["cyberbullying_type"] = imdf_cyber["cyberbullying_type"].replace({"ethnicity": 1})
imdf_cyber["cyberbullying_type"] = imdf_cyber["cyberbullying_type"].replace({"age": 1})
imdf_cyber["cyberbullying_type"] = imdf_cyber["cyberbullying_type"].replace({"gender": 1})
imdf_cyber["cyberbullying_type"] = imdf_cyber["cyberbullying_type"].replace({"religion": 1})
imdf_cyber["cyberbullying_type"] = imdf_cyber["cyberbullying_type"].replace({"other_cyberbullying": 1})
imdf_cyber.to_csv('Dataset/cyberbullying_tweets_class.csv', index=False)


# In[75]:


imdf_cyber.head()


# In[76]:


imdf_cyber.tail()


# ### 7. Cleaning the tweet text

# In[77]:


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


# In[78]:


imdf_cyber["tweet_text"] = imdf_cyber["tweet_text"].apply(clean_tweets)


# In[81]:


imdf_cyber.to_csv('Dataset/cyberbullying_tweets_remove.csv', index=False)


# In[82]:


imdf_cyber.head(30)


# ### 8. Drop empty tweets

# In[83]:


imdf_cyber= imdf_cyber[imdf_cyber['tweet_text'] != ""]


# In[84]:


imdf_cyber.to_csv('Dataset/cyberbullying_tweets_remove_empty.csv', index=False)


# In[85]:


imdf_cyber.head(30)


# In[86]:


imdf_cyber.info()


# ### 9. Graph for clean Dataset

# In[87]:


sns.set(rc={'figure.figsize':(10.7,7.7)})
sns.countplot(x='cyberbullying_type',data=imdf_cyber).set(title="Graph of clean Dataset")

