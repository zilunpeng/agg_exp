# coding: utf-8

# Copyright Zilun Peng 2017. You may use it under a Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License. See: http://creativecommons.org/licenses/by-nc-sa/4.0/deed.en
# Some code are taken from: http://www.cs.ubc.ca/~poole/cs532/2017/as1/Gender_from_ratings_mln.py and http://www.cs.ubc.ca/~poole/cs532/2017/as1/Gender_from_ratings_paper_version.py

# ## Load the files.
# To use this you need to download http://files.grouplens.org/datasets/movielens/ml-100k.zip
# See http://grouplens.org/datasets/movielens/
# Run this notebook (using "jupyter notebook" command) in the directory that contains the ml-100k directory.
#
# The following reads the ratings file and selects temporally first 60000 ratings.
# It trains on the users who were involved in first 40000 ratings.
# It tests on the other users who rated.

# In[1]:
import math, random

datasetname = "Yelp" # "100k"  # "1m"  # or "Yelp"
if datasetname=="100k":
    datafile, userfilename = "ml-100k/u.data","ml-100k/u.user"
    rating_cutoff, test_cutoff = 884673930, 880845177
elif datasetname=="1m":
    datafile, userfilename = "ml-1m/ratings.dat", "ml-1m/users.dat"
    rating_cutoff, test_cutoff = 974687810, 967587781
elif datasetname=="Yelp":
    datafile, usertrainfilename, usertestfilename = "new MC/yelp_mc_reviews.csv", "new MC/yelp_mc_class_train.csv","new MC/yelp_mc_class_test.csv"
else:
    assert False, ("not a valid dataset name",datasetname)

def extract_cols(lst,indexes):
    """extract sublist given by indexes from lst"""
    return (lst[i] for i in indexes)

with open(datafile,'r') as ratingsfile:
    if datasetname == "Yelp":
        ratings = [(int(rest[1:]), int(user[1:]), int(rating), 99999) for line in ratingsfile
                       for (user,rest,rating) in [tuple(line.strip().split(','))]   # for yelp
                             ]
        with open(usertrainfilename) as usertrainfile:
            gender_train = {int(rest[1:]): "F" if eth == "Mexican" else "M" for line in usertrainfile
                                for (rest,eth) in [tuple(line.strip().split(','))]
                                }
        with open(usertestfilename) as usertestfile:
            gender_test = {int(rest[1:]): "F" if eth == "Mexican" else "M" for line in usertestfile
                               for (rest,eth) in [tuple(line.strip().split(','))]
                                }
        training_users = set(gender_train)
        test_users = set(gender_test)
        all_users = training_users | test_users
        all_items = {(rest) for (usr,rest,r,tmp) in ratings}
        #assert all_users == {r for (r,u,i,d) in ratings}
    else:
        if datasetname == "100k":
            all_ratings = (tuple(int(e) for e in line.strip().split('\t'))   # for 100k
                             for line in ratingsfile)
        elif datasetname == "1m":
            all_ratings = (tuple(int(e) for e in extract_cols(line.strip().split(':'),[0,2,4,6])) # for 1m
                         for line in ratingsfile)
        ratings = [eg for eg in all_ratings if eg[3] <= rating_cutoff]
        all_users = {u for (u,i,r,d) in ratings}
        all_items = {i for (u, i, r, d) in ratings}
        print("There are ",len(ratings),"ratings and",len(all_users),"users")
        training_users = {u for (u,i,r,d) in ratings if d <= test_cutoff}
        test_users = all_users - training_users

        # extract the training and test dictionaries
        with open(userfilename,'r') as usersfile:
            if datasetname == "100k":
                user_info = (line.strip().split('|') for line in usersfile)
            elif datasetname == "1m":
                user_info = (extract_cols(line.strip().split(':'),[0,4,2,6,8]) for line in usersfile)
            gender_train, gender_test = {},{}
            for (u,a,g,o,p) in user_info:
                if int(u) in training_users:
                    gender_train[int(u)]=g
                elif int(u) in test_users:
                    gender_test[int(u)]=g

# check the results
assert len(training_users) == len(gender_train)
assert len(test_users) == len(gender_test)
print("There are ", len(gender_train), "users for training")
print("There are ", len(gender_test), "users for testing")
print("There are ", len(all_items), "items")

# In[2]:

# nf_tr = number in training set females
nf_tr = len({u for (u, g) in gender_train.items() if g == 'F'})
# tot_tr = total number of training users
tot_tr = len(gender_train)
print("Proportion of training who are female", nf_tr, '/', tot_tr, '=', nf_tr / tot_tr)


class Dataset(object):
    def __init__(self, name, gender_train, gender_test):
        self.name = name
        self.__str__ = name
        self.gender_train = gender_train
        self.gender_test = gender_test
        # nf_tr = number in training set females
        self.nf_tr = len({u for (u, g) in gender_train.items() if g == 'F'})
        # tot_tr = total number of training users
        self.tot_tr = len(gender_train)
        print("Proportion of training who are female", self.nf_tr, '/', self.tot_tr, '=', self.nf_tr / self.tot_tr)

        self.movie_stats = {}  # movie -> (#f, #m) dictionary
        for (u, i, r, d) in ratings:
            if u in gender_train:
                if i in self.movie_stats:
                    (nf, nm) = self.movie_stats[i]
                else:
                    (nf, nm) = (0, 0)
                if gender_train[u] == "F":
                    self.movie_stats[i] = (nf + 1, nm)
                if gender_train[u] == "M":
                    self.movie_stats[i] = (nf, nm + 1)


                    # ## Evaluation
                    # The following function can be used to evaluate your predictor on the test set.
                    # Your predictor may use ratings and gender_train but *not* gender_test. Your predictor should take a user and a second parameter called "para" that is a parameter that can be varied.

                    # In[5]:

    def evaluate(self, pred, **nargs):
        """pred is a function from users into real numbers that gives prediction P(u)='F',
        returns (sum_squares_error,  log loss)
        nargs should include para"""
        sse = sum((pred(u, ds=self, **nargs) - (1 if g == "F" else 0)) ** 2
                  for (u, g) in self.gender_test.items())
        ll = -sum(math.log(prn, 2) if g == 'F' else math.log(1 - prn, 2)
                  for (u, g) in self.gender_test.items()
                  for prn in [pred(u, ds=self, **nargs)])
        return (sse, ll, sse / len(self.gender_test), ll / len(self.gender_test))


evaluate_meaning = ["Sum Squares", "neg Log Likelihood", "Average Square Error", "Log Loss"]

original_ds = Dataset("all", gender_train, gender_test)

# ## Gradient descent for MLN

# In[5]:

import random
import math

def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def sigmoid_inv(s):
    return - math.log((1 - s) / s)


num_features = 5  # must be >= 1 (otherwise there is no signal)
features = range(num_features)
threshold = 4  # 2..5


def actual(rating):
    return 1 if rating >= threshold else 0


# user_to_rating_stats[u] = ( # positive ratings , # neg ratings)
user_pos_ratings_stats = {}
user_neg_ratings_stats = {}
for (user, item, rating, timestamp) in ratings:
    if user in user_pos_ratings_stats and rating >= threshold:
        user_pos_ratings_stats[user].append(item)
    elif user not in user_pos_ratings_stats and rating >= threshold:
        user_pos_ratings_stats[user] = [item]
    elif user in user_neg_ratings_stats and rating < threshold:
        user_neg_ratings_stats[user].append(item)
    elif user not in user_neg_ratings_stats and rating < threshold:
        user_neg_ratings_stats[user] = [item]

wgts_pos_r = [0]*max(all_items);
wgts_neg_r = [0]*max(all_items);
w0 = sigmoid_inv(original_ds.nf_tr / original_ds.tot_tr)  # initialize weight for G(U)
iter = 0


def pred_mln(user, ds=original_ds, para=lambda x: 0):
    prob = w0;
    if user in user_pos_ratings_stats:
        for pos_item in user_pos_ratings_stats[user]:
            prob = prob + wgts_pos_r[pos_item-1]
    if user in user_neg_ratings_stats:
        for neg_item in user_neg_ratings_stats[user]:
            prob = prob + wgts_neg_r[neg_item-1]

    return sigmoid(prob)


def learn(num_iter=20, ds=original_ds, step_size=1e-5, pregl=0, trace=True):
    global w0, w1, w2, iter, wgts_pos_r, wgts_neg_r
    sse, sll = 0, 0
    prev_sll = float("inf")
    training_users = tuple(ds.gender_train)
    for i in range(num_iter):
        sse, sll = 0, 0
        # old_weights = w0, w1, w2
        for user in random.sample(training_users, len(training_users)):
            error = pred_mln(user) - (1 if ds.gender_train[user] == "F" else 0)
            sse += error ** 2
            sll += -math.log(pred_mln(user) if ds.gender_train[user] == "F" else 1 - pred_mln(user), 2)
            w0 -= step_size * error
            if user in user_pos_ratings_stats:
                for pos_item in user_pos_ratings_stats[user]:
                    wgts_pos_r[pos_item-1] -= step_size * error
            if user in user_neg_ratings_stats:
                for neg_item in user_neg_ratings_stats[user]:
                    wgts_neg_r[neg_item-1] -= step_size * error
        w0 = w0 * (1 - pregl)
        wgts_neg_r = wgts_neg_r * (1 - pregl)
        wgts_pos_r = wgts_pos_r * (1 - pregl)
        iter += 1
        if trace:
            print("iteration", iter, "wts for G(U)=", w0, "ase=", sse / len(ds.gender_train), "all=",
                  sll / len(ds.gender_train))
        if prev_sll < sll:
            # step_size *= 0.5
            if trace: print("worsening step; step size", step_size)
            # w0,w1,w2 = old_weights
            prev_sll = sll
    print("after", iter, "iterations: evaluation=", ds.evaluate(pred_mln))


learn(60000, trace=False)