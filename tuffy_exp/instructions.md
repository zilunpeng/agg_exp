command for weight learning:

java -jar tuffy.jar -learnwt -i prog.mln -e mln_evidence.db -queryFile mln_query.db -r learnedWgtsMln.mln -mcsatSamples 5 -dMaxIter 5000

command for inference:

java -jar tuffy.jar -marginal -i learnedWgtsMln.mln -e mln_evidence.db -queryFile mln_query.db -r results.txt
