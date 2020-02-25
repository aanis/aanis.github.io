<a href="https://colab.research.google.com/github/lahorekid/ufo/blob/master/Twitter.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>


```python
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
import collections

import tweepy as tw
import nltk
from nltk.corpus import stopwords
import re
import networkx
from textblob import TextBlob

import warnings
warnings.filterwarnings("ignore")

sns.set(font_scale=1.5)
sns.set_style("whitegrid")
```


```python
consumer_key= 
consumer_secret= 
access_token= 
access_token_secret= 
```


```python
auth = tw.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tw.API(auth, wait_on_rate_limit=True)
```


```python
def remove_url(txt):

    return " ".join(re.sub("([^0-9A-Za-z \t])|(\w+:\/\/\S+)", "", txt).split())
# Create a custom search term and define the number of tweets
search_term = "#australia+fires -filter:retweets"

tweets = tw.Cursor(api.search,
                   q=search_term,
                   lang="en",
                   since='2020-01-01').items(1000)

# Remove URLs
tweets_no_urls = [remove_url(tweet.text) for tweet in tweets]
```


```python
# Create textblob objects of the tweets
sentiment_objects = [TextBlob(tweet) for tweet in tweets_no_urls]

sentiment_objects[0].polarity, sentiment_objects[0]
(-0.06666666666666665,
 TextBlob("Australian fires are ravaging the planet"))
```




    (-0.06666666666666665, TextBlob("Australian fires are ravaging the planet"))




```python
# Create list of polarity valuesx and tweet text
sentiment_values = [[tweet.sentiment.polarity, str(tweet)] for tweet in sentiment_objects]

sentiment_values[0]
```




    [0.033333333333333326,
     'Unknown facts about coronavirus infection The main source is Australia not China Those fires were not wast']




```python
# Create dataframe containing the polarity value and tweet text
sentiment_df = pd.DataFrame(sentiment_values, columns=["polarity", "tweet"])

sentiment_df.head()
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
      <th>polarity</th>
      <th>tweet</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.033333</td>
      <td>Unknown facts about coronavirus infection The ...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.000000</td>
      <td>Our hearts go out to those in Australia Austra...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-0.500000</td>
      <td>FlashbackFriday The BlackSaturday bush fires i...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.000000</td>
      <td>We had recent fires that burned out a lot of n...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-0.025000</td>
      <td>SamKimpton JoelGoldenberg1 JohnCleese Burnt fa...</td>
    </tr>
  </tbody>
</table>
</div>




```python
fig, ax = plt.subplots(figsize=(8, 6))

# Plot histogram of the polarity values
sentiment_df.hist(bins=[-1, -0.75, -0.5, -0.25, 0.25, 0.5, 0.75, 1],
             ax=ax,
             color="purple")

plt.title("Sentiments from Tweets on Australian Fires")
plt.show()
```


![png](output_8_0.png)


[Credit](https://www.earthdatascience.org/courses/earth-analytics-python/using-apis-natural-language-processing-twitter/analyze-tweet-sentiments-in-python/
)
