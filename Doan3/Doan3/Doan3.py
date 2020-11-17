
import pandas as pd
import numpy as np


read_data=pd.read_csv(r"C:\Users\hoang\Downloads\dulieu.csv")
print("Done.")

print('\n____________________________________ Dataset info ____________________________________')
print(read_data.info())              
print('\n____________________________________ Some first data examples ____________________________________')
print(read_data.head(6)) 
print('\n____________________________________ Statistics of numeric features ____________________________________')
print(read_data.describe())    


import re
# function for text cleaning 
def clean_text(text):
    # remove backslash-apostrophe 
    text = re.sub("\'", "", text) 

    # remove whitespaces 
    text = ' '.join(text.split()) 

    # remove everything except alphabets 
    # text = re.sub("[^a-zA-Z]"," ",text) 

    # convert text to lowercase 
    text = text.lower()
    
    text = re.findall(r'(?i)\b[a-záàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệóòỏõọôốồổỗộơớờởỡợíìỉĩịúùủũụưứừửữựýỳỷỹỵđ]+\b', text)

    return text

read_data['keyword'] = read_data['Mô tả'].apply(lambda x: clean_text(x))


import nltk
import seaborn as sns
import matplotlib.pyplot as plt 
def freq_words(x, terms = 30): 
  #all_words = ' '.join([text for text in x]) 
  all_words = ' '.join(str(text) for text in x)
  all_words = all_words.split() 
  fdist = nltk.FreqDist(all_words) 
  words_df = pd.DataFrame({'word':list(fdist.keys()), 'count':list(fdist.values())}) 
  
  # selecting top 20 most frequent words 
  d = words_df.nlargest(columns="count", n = terms) 
  
  # visualize words and frequencies
  plt.figure(figsize=(12,15)) 
  ax = sns.barplot(data=d, x= "count", y = "word") 
  ax.set(ylabel = 'Word') 
  plt.show()
  
# print 100 most frequent words 
freq_words(read_data['keyword'], 30)

nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

# function to remove stopwords
def remove_stopwords(text):
    no_stopword_text = [w for w in str(text).split() if not w in stop_words]
    return ' '.join(no_stopword_text)

read_data['keyword'] = read_data['keyword'].apply(lambda x: remove_stopwords(x))
freq_words(read_data['keyword'], 30)

# Location
from math import radians, cos, sin, asin, sqrt
def haversine(lon1, lat1, lon2, lat2):
       """
       Calculate the great circle distance between two points 
       on the earth (specified in decimal degrees)
       """
       # convert decimal degrees to radians 
       lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
       # haversine formula 
       dlon = lon2 - lon1 
       dlat = lat2 - lat1 
       a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
       c = 2 * asin(sqrt(a)) 
       # Radius of earth in kilometers is 6371
       km = 6371* c
       return km

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
count = CountVectorizer()
count_matrix = count.fit_transform(read_data['keyword'])
cosine_sim = cosine_similarity(count_matrix, count_matrix)
print(cosine_sim)

indices = pd.Series(read_data['Tiêu đề tin'])

def recommend(title, cosine_sim = cosine_sim):
    recommended_nhatrokeyword = []
    idx = indices[indices == title].index[0]
    score_series = pd.Series(cosine_sim[idx]).sort_values(ascending = False)
    top_10_indices = list(score_series.iloc[1:11].index)
    
    for i in top_10_indices:
        recommended_nhatrokeyword.append(list(read_data['Tiêu đề tin'])[i])
        
    return recommended_nhatrokeyword

print(recommend('Căn hộ chung cư Hưng Vượng 2 sang trọng'))