---
title: "NLP With Disaster Tweets"
date: 2020-03-03T23:32:31-06:00
summary: "Predict which Tweets are about real disasters and which ones are not"


reading_time: false  # Show estimated reading time?
share: false  # Show social sharing links?
profile: false  # Show author profile?
comments: false  # Show comments?

# Optional header image (relative to `static/img/` folder).
header:
  caption: "NLP With Disaster Tweets"
  image: "nlp1-cover.jpg"

---


```python
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import spacy
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
from sklearn.linear_model import LogisticRegression

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
```

    /kaggle/input/nlp-getting-started/test.csv
    /kaggle/input/nlp-getting-started/sample_submission.csv
    /kaggle/input/nlp-getting-started/train.csv
    

Loading the files.


```python
train = pd.read_csv("/kaggle/input/nlp-getting-started/train.csv")
test = pd.read_csv("/kaggle/input/nlp-getting-started/test.csv")
```

create vectors for all of our tweets.


```python
all_data = pd.concat([train,test], axis = 0, sort=False)
# Load the spacy model to get sentence vectors
nlp = spacy.load('en_core_web_lg')
vectors = np.array([nlp(tweet.text).vector for idx, tweet in all_data.iterrows()])
vectors.shape
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    <ipython-input-4-be4ac5812fb6> in <module>
    ----> 1 all_data = pd.concat([train,test], axis = 0, sort=False)
          2 # Load the spacy model to get sentence vectors
          3 nlp = spacy.load('en_core_web_lg')
          4 vectors = np.array([nlp(tweet.text).vector for idx, tweet in all_data.iterrows()])
          5 vectors.shape
    

    NameError: name 'train' is not defined



```python
# Center the vectors
vec_mean = vectors.mean(axis=0)
centered = pd.DataFrame([vec - vec_mean for vec in vectors])
```


```python

def svc_model(vectors, train):
    # Split train-validation data
    X_train, X_valid, y_train, y_valid = train_test_split(vectors[:len(train)], train.target, 
                                                          test_size=0.1, random_state=21)

    # Create the LinearSVC model
    model = LinearSVC(random_state=21, dual=False)
    
    # Fit the model
    model.fit(X_train, y_train)
    
    # Cross validation score over 3 folds
    scores = cross_val_score(model, X_train, y_train, cv=3)
    print("Cross validation over 3 folds: ", scores, " --- ", sum(scores)/3.)
    
    # Uncomment to see model accuracy
    #print(f'Model test accuracy: {model.score(X_valid, y_valid)*100:.3f}%')
    
    return model

model_svc_basic = svc_model(centered, train)
```


```python
# Submit results

y_test = model_svc_basic.predict(centered[-len(test):])
submission = pd.DataFrame({
    "id": test.id, 
    "target": y_test
})
submission.to_csv('submission_svc_basic.csv', index=False)
```
