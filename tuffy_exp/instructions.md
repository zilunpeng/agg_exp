1. Run mln_LR_tuffy_wgt_train.py to generate evidence and test files for weight learning

2. Run Tuffy to learn weight

        command for weight learning:
        java -jar tuffy.jar -learnwt -i prog.mln -e mln_evidence.db -queryFile mln_query.db -r learnedWgtsMln.mln -mcsatSamples 5 -dMaxIter 5000

3. Run mln_LR_tuffy_test.py to generate evidence and test files for inference

        command for inference:
        java -jar tuffy.jar -marginal -i learnedWgtsMln.mln -e mln_evidence.db -queryFile mln_query.db -r results.txt


4. Run mln_LR_tuffy_evalutate.py to generate the errors on results.txt
