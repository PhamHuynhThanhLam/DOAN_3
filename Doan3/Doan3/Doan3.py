
# In[0]: IMPORT
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import nltk
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

# In[1]: Xem nhanh dữ liệu 
read_data=pd.read_csv(r"C:\CNTT_Namw4_kỳ1\DOAN3\dulieu.csv")
print("Done.")


def clean(text):
    text = text.replace(";", ".") 
    return float(text)
read_data['Vĩ độ'] = read_data['Vĩ độ'].apply(lambda x: clean(x))
read_data['Kinh độ'] = read_data['Kinh độ'].apply(lambda x: clean(x))


print('\n____________________________________ Dataset info ____________________________________')
print(read_data.info())              
print('\n____________________________________ Some first data examples ____________________________________')
print(read_data.head(6)) 
print('\n____________________________________ Statistics of numeric features ____________________________________')
print(read_data.describe())    



# In[2]: Tìm key word 

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



def freq_words(x, terms = 30): 
  #all_words = ' '.join([text for text in x]) 
  all_words = ' '.join(str(text) for text in x)
  all_words = all_words.split() 
  fdist = nltk.FreqDist(all_words) 
  words_df = pd.DataFrame({'word':list(fdist.keys()), 'counts':list(fdist.values())}) 
  
  # selecting top 20 most frequent words 
  d = words_df.nlargest(columns="counts", n = terms) 
  
  # visualize words and frequencies
  plt.figure(figsize=(12,15)) 
  ax = sns.barplot(data=d, x= "counts", y = "word") 
  ax.set(ylabel = 'Word') 
  plt.show()
  
# in ra màn hình 30 dữ liệu nhiều nhất 
freq_words(read_data['keyword'], 30)

# Đã download
# nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

# function to remove stopwords
def remove_stopwords(text):
    no_stopword_text = [w for w in str(text).split() if not w in stop_words]
    return ' '.join(no_stopword_text)

read_data['keyword'] = read_data['keyword'].apply(lambda x: remove_stopwords(x))
freq_words(read_data['keyword'], 30)


count = CountVectorizer()
count_matrix = count.fit_transform(read_data['keyword'])
cosine_sim = cosine_similarity(count_matrix, count_matrix)
print(cosine_sim)


# In[3]: Tìm key location 
lon_min, lat_min, lon_max, lat_max = 10.01899606, 105.4282622 , 21.44498123, 109.2042187
nyc_events = read_data[(read_data['Kinh độ']>lon_min) & 
           (read_data['Kinh độ']<lon_max) & 
           (read_data['Vĩ độ']>lat_min) & 
           (read_data['Vĩ độ']<lat_max)]
nyc_events.head()
nyc_events.shape


from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
#kmeans = KMeans(n_clusters=70, init='k-means++')
## Compute the clusters based on longitude and latitude features
#X_sample = nyc_events[['Kinh độ','Vĩ độ']].sample(frac=0.1)
#kmeans.fit(X_sample)
#y = kmeans.labels_
#print("k = 70", " silhouette_score ", silhouette_score(X_sample, y, metric='euclidean'))

for k in range(20, 27, 10):
    kmeans = KMeans(n_clusters=k, init='k-means++')
    X_sample = (nyc_events[['Kinh độ','Vĩ độ']].sample(frac=0.1))
    kmeans.fit(X_sample)
    y = kmeans.labels_
    print("k =", k, " silhouette_score ", silhouette_score(X_sample, y, metric='euclidean'))

nyc_events['cluster'] = kmeans.predict(nyc_events[['Kinh độ','Vĩ độ']])
nyc_events[['Vĩ độ','Kinh độ','Tiêu đề tin','cluster']].sample(10)

gdf = nyc_events.groupby(['cluster', 'Tiêu đề tin']).size().reset_index()
gdf.columns = ['cluster', 'Tiêu đề tin', 'count']
idx = gdf.groupby(['cluster'])['count'].transform(max) == gdf['count']
topvenues_df = gdf[idx].sort_values(by='count', ascending=False)
#top 10 out of 200 clusters by events count
topvenues = topvenues_df[:20]

plt.style.use('ggplot')
fig = plt.figure()
fig.set_size_inches(21,5)

plt.bar(range(len(topvenues)),topvenues['Tiêu đề tin'], align='center')
plt.xticks(range(len(topvenues)), topvenues['count'])
plt.title('Most visited nhà trọ')
plt.show()


# In[4]: Tìm key word 
def recommend_venues(df, longitude, latitude):
    array = []
    predicted_cluster = kmeans.predict(np.array([longitude,latitude]).reshape(1,-1))[0]
    # Fetch the venue name of the top most record in the topvenues dataframe for the predicted cluster
    #venue_name = df[df['cluster']==predicted_cluster].iloc[0]['Tiêu đề tin']
    print(predicted_cluster)
    for i in range(0, len(df[df['cluster']==predicted_cluster])-1, 1):
        array.append(df[df['cluster']==predicted_cluster].iloc[i]['Tiêu đề tin'])
    #msg = 'What about visiting the ' + venue_name + '?'
    #return msg
    return array



indices = pd.Series(read_data['Tiêu đề tin'])
def recommend(title, cosine_sim = cosine_sim):
    recommended_nhatrokeyword = []
    idx = indices[indices == title].index[0]
    score_series = pd.Series(cosine_sim[idx]).sort_values(ascending = False)
    top_10_indices = list(score_series.iloc[1:11].index)
    
    for i in top_10_indices:
        recommended_nhatrokeyword.append(list(read_data['Tiêu đề tin'])[i])
        
    return recommended_nhatrokeyword


for i in recommend_venues(topvenues_df, 10.809097, 106.672583):
    print(i)
for i in recommend('CH Orchard Park View 95m² 3PN FUll NT cao cấp'):
    print(i)

temp = " "
for i in recommend_venues(topvenues_df, 10.809097, 106.672583):
    for j in recommend('CH Orchard Park View 95m² 3PN FUll NT cao cấp'):
        if i == j and temp == " " :
            temp = j;
            print(i)
            