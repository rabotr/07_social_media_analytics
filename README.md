

```python
# Dependencies
import tweepy
import numpy as np
from datetime import datetime
import pandas as pd
import collections
import matplotlib.pyplot as plt
import seaborn as sns

# Import and Initialize Sentiment Analyzer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()

# Twitter API Keys
from config import (consumer_key,
                    consumer_secret,
                    access_token,
                    access_token_secret)

# Setup Tweepy API Authentication
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth, parser=tweepy.parsers.JSONParser())
```


```python
# Target User Account
target_users = ("BBC", "CBS", "CNN", "FoxNews", "nytimes")

# dict for holding sentiments, users, text, and time
results_dict= collections.OrderedDict([("User", []),
                                       ("Text", []),
                                       ("Date", []),
                                       ("Compound", []),
                                       ("Positive", []),
                                       ("Neutral", []),
                                       ("Negative", []),
                                       ("Tweets Ago", [])])

# loop through target users
for user in target_users:
    
    count = 0
    
    # Loop through 5 pages of tweets (total 100 tweets)
    for x in range(1, 6):
        
        # Get all tweets from home feed
        public_tweets = api.user_timeline(user, page=x)
        
        # Loop through all tweets
        for tweet in public_tweets:         
            
            count = count + 1
            
            # Run Vader Analysis on each tweet
            results = analyzer.polarity_scores(tweet["text"])
            compound = results["compound"]
            pos = results["pos"]
            neu = results["neu"]
            neg = results["neg"]
            
            # store data
            results_dict["User"].append(user)
            results_dict["Text"].append(tweet["text"])
            results_dict["Date"].append(tweet["created_at"])
            results_dict["Compound"].append(results["compound"])
            results_dict["Positive"].append(results["pos"])
            results_dict["Neutral"].append(results["neu"])
            results_dict["Negative"].append(results["neg"]) 
            results_dict["Tweets Ago"].append(count)
```


```python
# create df
twitter_df = pd.DataFrame(results_dict)

twitter_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>User</th>
      <th>Text</th>
      <th>Date</th>
      <th>Compound</th>
      <th>Positive</th>
      <th>Neutral</th>
      <th>Negative</th>
      <th>Tweets Ago</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>BBC</td>
      <td>Would you describe yourself as a feminist?\n#H...</td>
      <td>Sat Jun 09 20:04:00 +0000 2018</td>
      <td>0.0000</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>BBC</td>
      <td>What is it like being Germaine Greer? This obs...</td>
      <td>Sat Jun 09 19:02:05 +0000 2018</td>
      <td>0.3612</td>
      <td>0.122</td>
      <td>0.878</td>
      <td>0.000</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>BBC</td>
      <td>The cast of @QueerEye brought a bag of goodies...</td>
      <td>Sat Jun 09 18:04:03 +0000 2018</td>
      <td>0.0000</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>BBC</td>
      <td>One hundred years after some women won the rig...</td>
      <td>Sat Jun 09 17:01:05 +0000 2018</td>
      <td>0.5719</td>
      <td>0.171</td>
      <td>0.829</td>
      <td>0.000</td>
      <td>4</td>
    </tr>
    <tr>
      <th>4</th>
      <td>BBC</td>
      <td>A year on from the Grenfell Tower fire, Sean a...</td>
      <td>Sat Jun 09 16:04:03 +0000 2018</td>
      <td>0.4019</td>
      <td>0.222</td>
      <td>0.623</td>
      <td>0.156</td>
      <td>5</td>
    </tr>
  </tbody>
</table>
</div>




```python
#export df to csv
twitter_df.to_csv("twitter_data.csv")
```


```python
# plot 1 - compound sentiments vs time by user
sns.lmplot( x="Tweets Ago", y="Compound", data=twitter_df, fit_reg=False, hue="User", legend=True)
plt.title("Compound Sentiments by User over Time (2018-06-09)")
plt.xlim([-2,102])
plt.show()
```


![png](output_4_0.png)



```python
# find overall sentiments
overall_df = pd.DataFrame(data= twitter_df.groupby('User')['Compound'].mean())
overall_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Compound</th>
    </tr>
    <tr>
      <th>User</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>BBC</th>
      <td>0.114459</td>
    </tr>
    <tr>
      <th>CBS</th>
      <td>0.320993</td>
    </tr>
    <tr>
      <th>CNN</th>
      <td>0.098543</td>
    </tr>
    <tr>
      <th>FoxNews</th>
      <td>0.064477</td>
    </tr>
    <tr>
      <th>nytimes</th>
      <td>0.082367</td>
    </tr>
  </tbody>
</table>
</div>




```python
# plot 2 - overall tweet sentiments by user
sns.barplot(x=["BBC", "CBS", "CNN", "FoxNews", "nytimes"], y="Compound", data=overall_df)
plt.title("Overall Sentiment of Tweets by User (2018-06-09)")
plt.ylim([-1,1])
plt.xlabel("Users")
plt.show()
```


![png](output_6_0.png)


#### Three observations:
1. CBS has the most positive twitter overall.
2. All the users have very positive and very negative tweet sentiments.
3. The overall sentiment of the users' tweets is almost neutral, despite observation 2.

It is hard to see if there is a trend over time in the tweet sentiment based on the visualizations used.

