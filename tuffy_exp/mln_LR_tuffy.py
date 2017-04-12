# coding: utf-8

# Copyright Zilun Peng 2017. You may use it under a Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License. See: http://creativecommons.org/licenses/by-nc-sa/4.0/deed.en
# Some code are copied from http://www.cs.ubc.ca/~poole/cs532/2017/as1/Gender_from_ratings_mln.py

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

with open("ml-100k/u.data", 'r') as ratingsfile:
    all_ratings = (tuple(int(e) for e in line.strip().split('\t'))
                   for line in ratingsfile)
    ratings = [eg for eg in all_ratings if eg[3] <= 884673930]
    all_users = {u for (u, i, r, d) in ratings}
    all_items = {i for (u, i, r, d) in ratings}
    print("There are ", len(ratings), "ratings and", len(all_users), "users")
    training_users = {u for (u, i, r, d) in ratings if d <= 880845177}
    test_users = all_users - training_users

# extract the training and test dictionaries
with open("ml-100k/u.user", 'r') as usersfile:
    user_info = (line.strip().split('|') for line in usersfile)
    gender_train, gender_test = {}, {}
    for (u, a, g, o, p) in user_info:
        if int(u) in training_users:
            gender_train[int(u)] = g
        elif int(u) in test_users:
            gender_test[int(u)] = g
    print("There are ", len(gender_train.keys()), "training users and", len(gender_test.keys()), "testing users")

with open("./tuffy_exp/mln_evidence.db", "w+") as mln_evi:
    for (u, i, r, d) in ratings:
        if  r >= 4 and u in gender_train:
            mln_evi.write("ratedGt(U"+str(u)+",I"+str(i)+")\n")
        if r < 4 and u in gender_train:
            mln_evi.write("ratedLt(U"+str(u)+",I"+str(i)+")\n")
    for u, g in gender_train.items():
        if g == 'F':
            mln_evi.write("gender(U" + str(u) + ")\n")
        if g == 'M':
            mln_evi.write("!gender(U" + str(u) + ")\n")

mln_evi.close()
# mln_que.close()
