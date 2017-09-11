---
layout: post
title:  "An Introduction to Running Predictive Models via Web Service"
excerpt: "This is a quick and dirty guide to setting up a model scoring web service in Python with Flask. Included is both a walkthrough, and gist containing the necessary code."
date:   2017-08-24 17:59:44 -0400
category: programming
tags: [python, flask, modeling]
---

Have you ever needed to score data with a predictive model in (near) real time, using a remote server/machine? While resources related to creating web services are plentiful, not many approach it from the angle of model scoring.

In this post, we're going to assume an understanding of predictive modeling and python, to focus on a simple way to set up a web service that can receive data and return model predictions over the internet. The post is divided into three parts:
  1. Model
  2. Listener (the web service) and
  3. Sender (to push data to the listener)

Individual files to run all three can be found in [this gist](https://gist.github.com/spencercarter/b4a16e9924a6dab46c9dd604860444d3) which is also linked in each section header.

*This guide is by no means definitive, but I hope you'll find the method straightforward and effective.*

___

<br>

# [Part 1: Model](https://gist.github.com/spencercarter/b4a16e9924a6dab46c9dd604860444d3#file-model_web_service_flask_post_p1-py)
To get a model going, let's just fit a simple LASSO with K-fold on the Boston housing data
This is a quick model just to get something running in our web service. It's roughly based on [This example code from sklearn](http://scikit-learn.org/stable/auto_examples/feature_selection/plot_select_from_model_boston.html#sphx-glr-auto-examples-feature-selection-plot-select-from-model-boston-py). 


```python
import sklearn as sk
import numpy as np
from sklearn.datasets import load_boston
from sklearn.linear_model import LassoCV
import pandas as pd
from flask import Flask
from flask import request
import requests
import pickle
from time import time

# Load up the Boston data... I like keeping things in pandas DFs
mass = load_boston()

# Lowecase the predictor names and send the data to Pandas
predictors = [var.lower() for var in mass.feature_names]
X = pd.DataFrame(mass['data'], columns=predictors)
y = pd.Series(mass['target']) # medv

# Run a 3-fold CV LASSO with 10 weights chosen by steepest descent
lasso = LassoCV(n_alphas=10, normalize=True, cv=3)

print("Fitting model...")
lasso.fit(X,y)

print("Model fit. R-Squared = {0:0.2f}%".format(100*lasso.score(X,y)))
```
Outputs:

    Fitting model...
    Model fit. R-Squared = 71.06%
    

This is enough for our demonstration... I'm assuming a basic familiarity with modeling concepts like variable selection/engineering and the use of test data, which are noticeable absent here.

Now, we're going to save the current objects down to disk so our listener and sender can access them. In the individual files, part 2 and 3 will begin with loading the objects in this pickle.


```python
# A couple simple pickle functions
def pickle_me(obj, outfile):
    with open(outfile, 'wb') as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    return None

def unpickle_me(infile):
    with open(infile, 'rb') as f:
        unpickled = pickle.load(f)
    return unpickled
               
to_pickle = {'X':X, 'lasso':lasso, 'dtypes':X.dtypes, 'predictors':predictors}
pickle_me(to_pickle, 'model_web_service.pickle')    
```

The model and data are ready. Let's set up the web service!

___
<br>
## [Part 2: Listener](https://gist.github.com/spencercarter/b4a16e9924a6dab46c9dd604860444d3#file-model_web_service_flask_post_p2-py)

Now that we've got a model, we're going to need two things to get out web service going:
1. A scoring function to turn data to predictions
2. A web service running on our local machine, listening for requests

___
<br>
### 2.1: Scoring Function
Let's start with the scoring function. This function is going to assume that the data will come in as a JSON, which we're just going to use to pass data in the form:

```
{<variable1>:<value>, <variable2>:<value>, ... , <variableN>:<value>}
```

This should look familiar; simple JSONs look like a dict in python. Just about any data structure we may want to impose will work here, but I've chosen JSON because it's easy to receive in the web service with *flask.request.get_json()*, and pandas makes it simple to send/score with *DataFrame*.to_json() and *pandas.read_json()*.

To ensure the variables are consistent with expectations, we're going to pass all the data received, but keep just the values that the model expects using the *.loc* slicer, and type them accordingly (object, int, float, etc.).

Let's take a look at what the JSON data will look like. Call *DataFrame*.to_json() on a slice of a DF.


```python
X.iloc[0].to_json()
```

Outputs:


    '{"crim":0.00632,"zn":18.0,"indus":2.31,"chas":0.0,"nox":0.538,"rm":6.575,"age":65.2,"dis":4.09,"rad":1.0,"tax":296.0,"ptratio":15.3,"b":396.9,"lstat":4.98}'



Pretty straightforward. Next, create a scoring function that takes a JSON, and returns a model prediction.


```python
def score_obs(indata, model=lasso, predictors=predictors, df_types=df_types):
    """
    Scoring function. Takes a JSON of variables and values, then runs them through the model and returns a prediction
    """

    # Structure the input into a DF, but then impose the expected variable structure, using the .loc slicer
    data = pd.read_json(indata,typ='series').to_frame().T.loc[:,predictors]
    
    # .loc will drop excess, and create variables that aren't present. Fill their values so the model runs
    data.fillna(0, inplace=True) # !!! In practice, you should actually build in error handling or imputation
    
    # Fix the dtypes
    for c in data:
        data[c] = data[c].astype(df_types[c])
    
    # predict returns an array, so grab the first value
    return model.predict(data.values.reshape(1,-1))[0]

# To make sure everything is working, compare a direct prediction against one from the function
print("Direct prediction: {} \nUsing this function: {}".format(lasso.predict(X.iloc[0:1])[0],score_obs(X.iloc[0].to_json())))
```

Outputs:

    Direct prediction: 30.505530469875787 
    Using this function: 30.505530469875787
    

*A note on some hangups I've run into:*

As you can see in the above, *to_json()* returns a string. To read a single observation with *read_json()*, we need to: 

```python
pd.read_json(indata,typ='series')
```
Read it as a series, shaped (p,). Then,

```python
.to_frame().T
```

Sends that series to a DataFrame, and transposes to get the (1,p) shape our model will expect.

Now the model is in memory, and we have a scoring function! Let's set up the web service.

___
<br>
### 2.2: Web Service

This is a simple [RESTful](https://en.wikipedia.org/wiki/Representational_state_transfer) web service that accepts data via POST, and returns the model prediction.

*Strictly speaking, model predictions may more closely match GET requests in REST, because they can be repeated without care and don't alter anything on the server (idempotent). However, I'm using POST because model data can get extremely lengthy, and might exceed what some browsers can handle in a URL. [Here's some more info about REST](http://blog.teamtreehouse.com/the-definitive-guide-to-get-vs-post).*


```python
app = Flask(__name__)

app.silent = True # Suppresses logging and errors. In practice, you'll likely want them, or import logging
 
@app.route("/mass", methods=['POST'])
def predict_medv():
    '''Get the JSON from the request and send to our scoring function'''
    score = score_obs(request.get_json()) 
    return "{0:0.2f}".format(score)

app.run(port=1234) # defaults to localhost. Use host= option to change
```

If you turn off *silent*, it outputs:


     * Running on http://127.0.0.1:1234/ (Press CTRL+C to quit)
    
By default, Flask will print to the log every time a request is made to the web service. I'm disabling it here to reduce output volume, but in practice, logging is probably a good idea. It's also possible to set the "level" of event you want to log (like only errors). 
	
Let's break the key pieces of Flask down:

```python
@app.route("/mass", methods=['POST'])
```

Configures the the web service to run at ```<host>:<port>/mass``` and specifies that we'll be sending POST requests.

```python
def predict_medv():
```

Creates the function that will execute upon requests. Whatever it returns is what will be sent back to the requester. Here, we're just returning the prediction. And lastly,

```python
app.run(port=1234)
```

Starts up the web service on the ```<host>:<port>``` specified, defaulting host to localhost (127.0.0.1).

At this point, the web service is up, and will keep the shell running; if you're in a notebook, this will lock it up. In the raw code, I've broken out the listener from the sender so you can run both at the same time.

___
<br>
## [Part 3: Sender](https://gist.github.com/spencercarter/b4a16e9924a6dab46c9dd604860444d3#file-model_web_service_flask_post_p3-py)

Home stretch! Now we just need a way to send data to our web service.

We'll to generate test cases from our data, passed as a JSON. To make things a little simpler here, we're going to make a sender that takes a row number, and sends a JSON of the data to our web service.

```python
def make_a_post(indx):
    url = "http://127.0.0.1:1234/mass"
    data = X.iloc[indx].to_json()
    req = requests.post(url, json=data)
    if req.status_code != 200:
        return .0
    return float(req.content.decode("utf-8")) # The prediction will come in as a byte. Decode, then make float
```

To break a couple things down here:

```python
req = requests.post(url, json=data)
```
Makes the POST request to the web service, passing in out JSON of data. And,

```python
if req.status_code != 200:
	return .0
```

Is included because web requests use a status code to determine success (you've probably seen "404 Not Found" before). Status code 200 indicates success, and all others indicate something failed. As a simple failsafe, we'll return 0.0 to signal a failure.

Finally, let's toss observation 0 to the web service. We know from earlier, that it should return 30.505530469875787, rounded to 2 decimal-places. 


```python
make_a_post(0)
```

Returns:

    30.51

And a few more for show. Let's sample over the observations:


```python
for i in np.random.choice(range(X.shape[0]), 10):
    print("Obs {0:03d}: {1}".format(i, make_a_post(i)))
```

Outputs: 

    Obs 257: 41.79
    Obs 195: 37.82
    Obs 166: 37.75
    Obs 168: 26.51
    Obs 027: 16.4
    Obs 289: 26.16
    Obs 107: 20.25
    Obs 225: 38.77
    Obs 322: 23.24
    Obs 504: 26.59
    

And lastly, let's see how long these take to run. Determine the average of 100 runs:


```python
lag = []
rep = 100
for _ in range(100):
    t0 = time()
    make_a_post(0)
    lag.append(time()-t0)
print("Over {0} runs, the web service averaged {1:0.4f} seconds to score".format(rep,np.mean(lag)))
```

Outputs:

    Over 100 runs, the web service averaged 0.0114 seconds to score
    
*Of course, being on the local machine makes things quick, but I've typically seen sub-second return times on remote machines too.*

**Success!** We now have a web service running a model, that will return predictions remotely through an HTTP request.

Sky's the limit here. This post focused only on cases where you might stream individual data. However, Flask can also receive files, allowing us to send a CSV containing multiple observations to score. Maybe a topic for a future post.

To learn more about flask, check out the <a href="http://flask.pocoo.org/">Official Documentation</a> and the <a href="https://pythonprogramming.net/practical-flask-introduction/">pythonprogramming.net tutorial</a>.
