from flask import request, jsonify
from flask import send_file
import flask as Flask
from flask_cors import CORS, cross_origin
from flask_restful import Resource, Api, reqparse
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tweepy
import kapi
from tweepy import OAuthHandler
from textblob import TextBlob
import nltk
import string
import re
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
nltk.download('omw-1.4')
lemmatizer = WordNetLemmatizer()
from wordcloud import WordCloud, STOPWORDS

app = Flask.Flask(__name__)
CORS(app)
api = Api(app)
@app.route('/var', methods = ['GET'])
def var():
    args = request.args
    topicname = args.get('topicname',default = "Imran Khan", type = str)
    count = args.get('count', default = 5, type = int)
    consumerKey,consumerKeySecret,accessToken,accessTokenSecret=kapi.newaccess()
    auth = OAuthHandler(consumerKey, consumerKeySecret)
    auth.set_access_token(accessToken, accessTokenSecret)
    api = tweepy.API(auth)
    print(api)
    df = pd.DataFrame(columns=["Date","User","IsVerified","Tweet","Likes","RT",'User_location'])
    print(df)
    def get_tweets(Topic,Count):
        i=0

        for tweet in tweepy.Cursor(api.search_tweets, q=Topic,count=500, lang="en",tweet_mode='extended',result_type='recent').items():
            print(i, end='\r')
            date=tweet.created_at
            df.loc[i,"Date"] = date.strftime("%Y-%m-%d")
            df.loc[i,"User"] = tweet.user.name
            df.loc[i,"IsVerified"] = tweet.user.verified
            df.loc[i,"Tweet"] = tweet.full_text
            df.loc[i,"Likes"] = tweet.favorite_count
            df.loc[i,"RT"] = tweet.retweet_count
            df.loc[i,"User_location"] = tweet.user.location
            #df.to_csv("TweetDataset.csv",index=False)
            df.to_excel('{}.xlsx'.format("TweetDataset"),index=False)   ## Save as Excel
            i=i+1
            if i>Count:
                break
            else:
                pass
        # take input from user
    Topic = topicname+" -filter:retweets"
    # Topic=["Pakistan"]
    get_tweets(Topic , count)

    # Showing the data the Data
    df.head(10)
    def clean_tweet(tweet):
        # tokenization
        tweet = word_tokenize(tweet)
        # lower case
        tweet = [word.lower() for word in tweet]
        tweet = [lemmatizer.lemmatize(word) for word in tweet]
        tweet = [word for word in tweet if word.isalpha()]
        tweet = " ".join(tweet)
        tweet = tweet.strip()
        tweet = tweet.translate(str.maketrans('', '', string.punctuation))
        # remove http links
        tweet = re.sub(r"http\S+", "", tweet)
        tweet = re.sub(r"@\S+", "", tweet)
        tweet = re.sub(r"[^\w\s]", "", tweet)
        tweet = re.sub(r"\d+", "", tweet)
        # remove https and username and special characters
        tweet = ' '.join(re.sub("(#[A-Za-z0-9]+)|(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)"," ",tweet).split())
        tweet = tweet.split()
        tweet = [word for word in tweet if word not in stopwords.words('english')]
        tweet = " ".join(tweet)
        tweet = PorterStemmer().stem(tweet)
        tweet = tweet.strip('https')
        tweet =tweet.split()
        tweet = [t for t in tweet if len(t) > 2]
        tweet = " ".join(tweet)
        return tweet


    def analyze_sentiment(tweet):
        analysis = TextBlob(tweet)
        if analysis.sentiment.polarity >=0 and analysis.sentiment.polarity <=0.3:
            some_positive = "some positive"
            return some_positive
        elif analysis.sentiment.polarity >=0.3 and analysis.sentiment.polarity <=0.6:
            positive = "positive"
            return positive
        elif analysis.sentiment.polarity >=0.6 and analysis.sentiment.polarity <=1:
            very_positive = "very positive"
            return very_positive
        elif analysis.sentiment.polarity >=-0.3 and analysis.sentiment.polarity <=0:
            some_negative = "some negative"
            return some_negative
        elif analysis.sentiment.polarity >=-0.6 and analysis.sentiment.polarity <=-0.3:
            negative = "negative"
            return negative
        elif analysis.sentiment.polarity >=-1 and analysis.sentiment.polarity <=-0.6:
            very_negative = "very negative"
            return very_negative
        else:
            neutral = "neutral"
            return neutral


    def prepCloud(Topic_text,Topic):
        Topic = str(Topic).lower()
        Topic=' '.join(re.sub('([^0-9A-Za-z \t])', ' ', Topic).split())
        Topic = re.split("\s+",str(Topic))
        stopwords = set(STOPWORDS)
        stopwords.update(Topic) ### Add our topic in Stopwords, so it doesnt appear in wordClous
        ###
        text_new = " ".join([txt for txt in Topic_text.split() if txt not in stopwords])
        return text_new
    prepCloud(Topic,Topic)# chk krna hai isy
    #.......................>
    df['clean_tweet'] = df['Tweet'].apply(lambda x : clean_tweet(x))
    df.head(5)
    #.......................>
    df["Sentiment"] = df["Tweet"].apply(lambda x : analyze_sentiment(x))
    df.head(10)
    df.to_excel('{}.xlsx'.format("Sentiment"),index=False)

    #.......................>
    # ................ .............. .............. ............ >
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.pipeline import Pipeline
    from sklearn.svm import LinearSVC
    from sklearn.linear_model import LogisticRegression
    import numpy as np

    x = df['clean_tweet']
    y = df['Sentiment']
    x, x_test, y, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    vec = CountVectorizer()
    x = vec.fit_transform(x)
    x_test = vec.transform(x_test)
    model = MultinomialNB()
    model.fit(x, y)
    model.score(x_test, y_test)
    print("Accuracy of Naive Bayes Classifier is : ",model.score(x_test, y_test)*100,"%")
    accuracy = model.score(x_test, y_test)*100


    # print report of sentiment analysis
    # plt.figure(figsize=(10, 5))
    # plt.title('Sentiment Analysis')
    # plt.xlabel('Sentiment')
    # plt.ylabel('Counts')
    # plt.hist(df['Sentiment'])
    # plt.show()
    # # print pie chart of sentiment analysis
    # plt.figure(figsize=(10, 5))
    # plt.title('Sentiment Analysis')
    # plt.pie(df['Sentiment'].value_counts(), labels=df['Sentiment'].value_counts().index, autopct='%1.1f%%')

    # plot word cloud for maximum occuring words
    # wordcloud = WordCloud(width = 800, height = 800)
    # wordcloud.generate(prepCloud(df['clean_tweet'].str.cat(sep=' '),Topic))
    # wordcloud.to_image()


    #,,,,,,,,,,,,,
    plt.figure(figsize=(10,10))
    # print user location graph for first 100 tweets
    data = pd.read_excel('Sentiment.xlsx')
    data = data.head(100)
    # User location with sentiment
    sns.countplot(y='User_location',hue='Sentiment',data=data)
    #.............
    plt.savefig('graph.png', bbox_inches='tight')
    import base64
    with open("graph.png", "rb") as image_file:encoded_string = base64.b64encode(image_file.read())
    imagebytes = encoded_string.decode('utf-8')




    # ................ .............. .............. ............ >





    # This returns a dictionary of the POST request payload
    data = {}
    data = {
            "Date": df["Date"].to_dict(),
            "User": df["User"].to_dict(),
            "Tweetslocation": df["User_location"].to_dict(),
            "CleanTweets" : df["clean_tweet"].to_dict(),
             "Polarity" : df["Sentiment"].to_dict(),
             "OrignalTweets" : df["Tweet"].to_dict(),
             "TotalTweets" : count,
             "Accuracy" : accuracy,
             "Image" : imagebytes,
    }
    return data






# For Running our Api on Localhost
if __name__ == '__main__':
    # app.run(debug=True)
    app.run()   
    

