
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
from sklearn.naive_bayes import BernoulliNB

# In[1]: Xem nhanh dữ liệu 
motels = pd.read_csv(r"D:\HK1-2020-2021\Đồ án 3\DULIEU\dulieu.csv")
moteltest = pd.read_csv(r"D:\HK1-2020-2021\Đồ án 3\DULIEU\dulieutest.csv")
## ĐỔI DType of GIÁ thành float  
#def clean1(text):
#    text = text.replace(",", ".") 
#    return float(text)
#motels['Giá'] = motels['Giá'].apply(lambda x: clean1(x))
#motels['Giá'] = motels['Giá'].astype('float')
#print("Done.")

# Chuyển data số khoảng cách
def clean(text):
    text = text.replace(";", ".") 
    return float(text)
motels['Vĩ độ'] = motels['Vĩ độ'].apply(lambda x: clean(x))
motels['Kinh độ'] = motels['Kinh độ'].apply(lambda x: clean(x))

moteltest['Vĩ độ'] = moteltest['Vĩ độ'].apply(lambda x: clean(x))
moteltest['Kinh độ'] = moteltest['Kinh độ'].apply(lambda x: clean(x))

# In[2]: Load data training set
print('\n____________________________________ Dataset info ____________________________________')
print(motels.info())              
print('\n____________________________________ Some first data examples ____________________________________')
print(motels.head(6)) 
print('\n____________________________________ Statistics of numeric features ____________________________________')
print(motels.describe())    
 

# In[3]: Tách nội dung thành cách từ kháo keyword
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
motels['merger table'] = motels['Tiêu đề tin']+' '+motels['Mô tả']
motels['keyword'] = motels['merger table'].apply(lambda x: clean_text(x))
moteltest['keyword'] = (motels['Tiêu đề tin']+' '+motels['Mô tả']).apply(lambda x: clean_text(x))

# Dùng stopword xóa các từ ảnh hưởng và không liên quan
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
  
# In ra màn hình 30 dữ liệu nhiều nhất 
freq_words(motels['keyword'], 30)

# Đã download
# nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

# function to remove stopwords
def remove_stopwords(text):
    no_stopword_text = [w for w in str(text).split() if not w in stop_words]
    return ' '.join(no_stopword_text)

motels['keyword'] = motels['keyword'].apply(lambda x: remove_stopwords(x))
freq_words(motels['keyword'], 30)
moteltest['keyword'] = moteltest['keyword'].apply(lambda x: remove_stopwords(x))

count = CountVectorizer()
count_matrix = count.fit_transform(motels['keyword'])
count_matrix_test = count.fit_transform(moteltest['keyword'])

# Đánh số label
# Training
train_data = count_matrix.toarray()
label = np.random.randint(5, size=(1000))

# Call MultinomialNB
clf = BernoulliNB()
clf.fit(train_data, label)
array = clf.predict(train_data)
motels['labelrank'] = pd.DataFrame(array, columns=['labelrank'])

# In[4]: Similarity between motels
from scipy import spatial

def Similarity(motelid1, motelid2):
    wordsA = count_matrix.toarray()[motelid1]
    wordsB = count_matrix.toarray()[motelid2]
    wordsDistance = spatial.distance.cosine(wordsA, wordsB)
    return wordsDistance

Similarity(7,100)

print(motels.iloc[7])
print(motels.iloc[100])


# In[5]: Score Predictor
import operator
def predict_score(title, id):
    #name = input('Nhập title của nhà trọ: ')
    #title = "THIẾT KẾ MỚI - NỘI THẤT ĐẸP - 2PN 65m2 - SAFIRA Q9"
    name = title; 
    try:
        new_motel = motels[motels['Tiêu đề tin'].str.contains(name)].iloc[0].to_frame().T
    
        original_title = 'Tiêu đề tin';
        print('Selected Movie: ',new_motel['Tiêu đề tin'].values[0])
        array = np.array([count_matrix_test.toarray()[id]])
        print('Label key: ',clf.predict(array))
        def getNeighbors(baseMovie, K):
            distances = []
    
            for index, motel in motels.iterrows():
                if motel['STT'] != baseMovie['STT'].values[0]:
                    dist = Similarity(baseMovie['STT'].values[0], motel['STT'] - 1)
                    distances.append((motel['STT'], dist))
    
            distances.sort(key=operator.itemgetter(1))
            neighbors = []
    
            for x in range(K):
                neighbors.append(distances[x])
            return neighbors


        # Lấy 10 motel
        K = 10
        labelkey = 0
        neighbors = getNeighbors(new_motel, K)
        print('\nRecommended Movies: \n')
        for neighbor in neighbors:
            labelkey = labelkey + motels.iloc[neighbor[0]][16]
            print("{} ,Similarity = {} ,labelrank = {}".format(motels.iloc[neighbor[0]][1], neighbor[1], motels.iloc[neighbor[0]][16]))
    
        print('\n')

        labelkey = labelkey/K
        print('The predicted for %s là: %f' %(new_motel['Tiêu đề tin'].values[0],labelkey))
        print('The actual for %s là %f' %(new_motel['Tiêu đề tin'].values[0],new_motel['labelrank']))
        mse = float(new_motel['labelrank'].values[0])/labelkey
        rmse = np.sqrt(mse)
        print('MSE = {}'.format(mse))
        print('RMSE = {}'.format(rmse))
    except:
        print("Không tìm thể tìm thấy")
    
  

# In[6]: 1000 query
row_count = len(moteltest.axes[0]) # count row
for numberrow in range(0, 2 , 1):
    print("Query {}".format(numberrow));
    predict_score(moteltest['Tiêu đề tin'][numberrow],moteltest['STT'][numberrow]);
    print("/n")
