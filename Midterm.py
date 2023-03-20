# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import requests 
import urllib.request
import time
from bs4 import BeautifulSoup
from wordcloud import WordCloud
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer 

# Documenting the URLs of Tesla's sites
Tesla_URLs = ["https://www.tesla.com/", 
              "https://twitter.com/Tesla", 
              "https://www.instagram.com/teslamotors/", 
              "https://www.youtube.com/user/TeslaMotors",
              "https://www.facebook.com/TeslaMotors"]

# Scraping all the user-generated comments and data from their weblogs and social media platforms
# Storing the data in a .csv file
user_comments = []
for url in Tesla_URLs:
    # Scraping the page
    page = requests.get(url)
    soup = BeautifulSoup(page.content, 'html.parser')
    # Extracting all the user comments
    comments = soup.find_all('div', {'class': 'user-comment'})
    # Looping over the comments
    for comment in comments:
        user_comments.append(comment.text)

# Creating a dataframe
df = pd.DataFrame(user_comments, columns=['comment']) 

# Saving the dataframe to a csv file
df.to_csv('Tesla_user_comments.csv', index=False)

# Creating a Jupyter notebook and importing necessary libraries

# Loading your social media data
df = pd.read_csv('Tesla_user_comments.csv')

# Cleaning the text data

# Removing URLs
df['comment'] = df['comment'].str.replace('http\S+|www.\S+', '', case=False)

# Removing special characters
df['comment'] = df['comment'].str.replace('[^\w\s]', '')

# Converting text to lowercase
df['comment'] = df['comment'].str.lower()

# Tokenizing the text data
tokenizer = RegexpTokenizer(r'\w+')
df['comment'] = df['comment'].apply(lambda x: tokenizer.tokenize(x))

# Removing stopwords
stop_words = set(stopwords.words('english'))
df['comment'] = df['comment'].apply(lambda x: [word for word in x if word not in stop_words])

# Stemming and displaying a sample of the stemmed data
porter = PorterStemmer()
df['comment'] = df['comment'].apply(lambda x: [porter.stem(word) for word in x])
print(df['comment'].sample(5))

# Lemmatizing and displaying a sample of the lemmatized data
lemma = WordNetLemmatizer()
df['comment'] = df['comment'].apply(lambda x: [lemma.lemmatize(word) for word in x])
print(df['comment'].sample(5))

# Using word cloud to visualize the data
all_words = ' '.join([text for text in df['comment']])
wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(all_words)

plt.figure(figsize=(10, 7))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.show()

# Saving the cleaned data to a new CSV file
df.to_csv('Tesla_user_comments_cleaned.csv', index=False)

