#### 100k dataset

weights:

    0.4623        gender(v0)  //1.0
    0.0089       !ratedGt(v0, v1)  v  gender(v0)  //2.0
    0.0905       !ratedLt(v0, v1)  v  gender(v0)  //3.0
    
    0.8303        gender(v0)  //1.0
    -0.0036       !ratedGt(v0, v1)  v  gender(v0)  //2.0
    0.0341       !ratedLt(v0, v1)  v  gender(v0)  //3.0
    
    0.9076        gender(v0)  //1.0
    0.1351       !ratedGt(v0, v1)  v  gender(v0)  //2.0
    0.1912       !ratedLt(v0, v1)  v  gender(v0)  //3.0

    0.4444        gender(v0)  //1.0
    0.0514       !ratedGt(v0, v1)  v  gender(v0)  //2.0
    0.1395       !ratedLt(v0, v1)  v  gender(v0)  //3.0
    
    0.2469        gender(v0)  //1.0
    0.016       !ratedGt(v0, v1)  v  gender(v0)  //2.0
    0.0203       !ratedLt(v0, v1)  v  gender(v0)  //3.0


results:

    tuffy result on  100k  is (113.9373, inf, 0.6663, inf)
    tuffy result on  100k  is (107.92640000000002, inf, 0.631148538011696, inf)
    tuffy result on  100k  is (116.83339999999993, inf, 0.6832362573099411, inf)
    tuffy result on  100k  is (116.45749999999995, inf, 0.6810380116959062, inf)
    tuffy result on  100k  is (100.53280000000004, inf, 0.5879111111111114, inf)


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
    tuffy result on  1m  is (413.2130000000014, 1604.1444653006183, 0.2918170903954812, 1.1328703850993067)
    tuffy result on  1m  is (719.2526999999994, inf, 0.507946822033898, inf)
    tuffy result on  1m  is (365.7346999999991, 1450.3648912377532, 0.2582872175141237, 1.0242689909871139)


#### yelp dataset 

weights:
    
    -0.0001        gender(v0)  //1.0
    0.0013       !ratedGt(v0, v1)  v  gender(v0)  //2.0
    -0.0021       !ratedLt(v0, v1)  v  gender(v0)  //3.0
    
    0.0002        gender(v0)  //1.0
    0.002       !ratedGt(v0, v1)  v  gender(v0)  //2.0
    -0.0015       !ratedLt(v0, v1)  v  gender(v0)  //3.0
    
    0.0003        gender(v0)  //1.0
    0.0009       !ratedGt(v0, v1)  v  gender(v0)  //2.0
    -0.0014       !ratedLt(v0, v1)  v  gender(v0)  //3.0
    
    -0.0007        gender(v0)  //1.0
    0.0014       !ratedGt(v0, v1)  v  gender(v0)  //2.0
    -0.0014       !ratedLt(v0, v1)  v  gender(v0)  //3.0
    
    0.0002        gender(v0)  //1.0
    0.0003       !ratedGt(v0, v1)  v  gender(v0)  //2.0
    -0.0019       !ratedLt(v0, v1)  v  gender(v0)  //3.0
    
results:
    
    tuffy result on  Yelp  is (221.98710000000028, 886.3591165435612, 0.251686054421769, 1.0049423090063052)
    tuffy result on  Yelp  is (226.51290000000003, 899.5713956481897, 0.25681734693877556, 1.019922217288197)
    tuffy result on  Yelp  is (222.91700000000012, 889.098397003952, 0.2527403628117915, 1.0080480691654785)
    tuffy result on  Yelp  is (221.65200000000019, 885.4087606246848, 0.2513061224489798, 1.0038648079644952)
    tuffy result on  Yelp  is (225.47570000000027, 896.5957097202127, 0.25564138321995494, 1.016548423719062)








