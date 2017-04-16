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
datasetname = "1m" # "100k"  # "1m"  # or "Yelp"
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
            gender_train = {int(rest[1:]):"F" if eth=="Mexican" else "M" for line in usertrainfile
                                for (rest,eth) in [tuple(line.strip().split(','))]
                                }
        with open(usertestfilename) as usertestfile:
            gender_test = {int(rest[1:]):"F" if eth=="Mexican" else "M" for line in usertestfile
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



#Write Tuffy program

with open("./tuffy_exp/mln_evidence.db", "w") as mln_evi:
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

with open("./tuffy_exp/mln_query.db", "w") as mln_que:
    mln_que.write("gender(x)")

mln_evi.close()
mln_que.close()