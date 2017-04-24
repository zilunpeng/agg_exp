#### 1m dataset

weights:

    0.4459        gender(v0)  //1.0
    0.0164       !ratedGt(v0, v1)  v  gender(v0)  //2.0
    0.0189       !ratedLt(v0, v1)  v  gender(v0)  //3.0

    0.213        gender(v0)  //1.0
    -0.0004       !ratedGt(v0, v1)  v  gender(v0)  //2.0
    0.0091       !ratedLt(v0, v1)  v  gender(v0)  //3.0

    0.1517        gender(v0)  //1.0
    -0.0014       !ratedGt(v0, v1)  v  gender(v0)  //2.0
    0.0043       !ratedLt(v0, v1)  v  gender(v0)  //3.0

    0.2042        gender(v0)  //1.0
    0.0061       !ratedGt(v0, v1)  v  gender(v0)  //2.0
    0.0092       !ratedLt(v0, v1)  v  gender(v0)  //3.0
    
    0.063        gender(v0)  //1.0
    -0.001       !ratedGt(v0, v1)  v  gender(v0)  //2.0
    0.001       !ratedLt(v0, v1)  v  gender(v0)  //3.0

results:

    tuffy result on  1m  is (762.3861999999996, inf, 0.538408333333333, inf)
    tuffy result on  1m  is (494.9413000000022, inf, 0.3495348163841823, inf)
    tuffy result on  1m  is (413.2130000000014, -1604.1444653006183, 0.2918170903954812, -1.1328703850993067)
    tuffy result on  1m  is (719.2526999999994, inf, 0.507946822033898, inf)
    tuffy result on  1m  is (365.7346999999991, -1450.3648912377532, 0.2582872175141237, -1.0242689909871139)

#### yelp dataset 

weights:

    0.0001        gender(v0)  //1.0
    -0.0102       !ratedGt(v0, v1)  v  gender(v0)  //2.0
    -0.0032       !ratedLt(v0, v1)  v  gender(v0)  //3.0
    
    0.0003        gender(v0)  //1.0
    -0.0442       !ratedGt(v0, v1)  v  gender(v0)  //2.0
    -0.0085       !ratedLt(v0, v1)  v  gender(v0)  //3.0
    
results:

    tuffy result on  Yelp  is (237.2989999999999, -935.9786945303259, 0.26904648526077085, -1.0612003339346099)
    tuffy result on  Yelp  is (272.2687000000001, inf, 0.30869467120181415, inf)




