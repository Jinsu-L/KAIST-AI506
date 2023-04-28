STUDENT_ID = 20223558
all: all_py # or all_py

wordcount_java:
	cd wordcount_java_files && hadoop com.sun.tools.javac.Main WordCount.java
	cd wordcount_java_files && jar cf wc.jar WordCount*.class
	hadoop fs -rm -r -f /user/$(STUDENT_ID)/wc > /dev/null
	hadoop fs -mkdir /user/$(STUDENT_ID)/wc > /dev/null
	hadoop fs -put /hadoop/*.txt /user/$(STUDENT_ID)/wc > /dev/null
	cd wordcount_java_files && yarn jar wc.jar WordCount /user/$(STUDENT_ID)/wc /user/$(STUDENT_ID)/wc/output
	hadoop fs -cat /user/$(STUDENT_ID)/wc/output/part-r-00000
	hadoop fs -rm -r -f /user/$(STUDENT_ID)/wc > /dev/null
wordcount_py:
	hadoop fs -rm -r -f /user/$(STUDENT_ID)/wc > /dev/null
	hadoop fs -mkdir /user/$(STUDENT_ID)/wc > /dev/null
	hadoop fs -put /hadoop/*.txt /user/$(STUDENT_ID)/wc > /dev/null
	mapred streaming -files ./wordcount_py_files -input /user/$(STUDENT_ID)/wc -output /user/$(STUDENT_ID)/wc/output -mapper ./wordcount_py_files/mapper.py -reducer ./wordcount_py_files/reducer.py > /dev/null
	hadoop fs -cat /user/$(STUDENT_ID)/wc/output/part-00000
	hadoop fs -rm -r -f /user/$(STUDENT_ID)/wc > /dev/null
all_java:
	cd lsh_java_files && hadoop com.sun.tools.javac.Main LSH.java
	cd lsh_java_files && jar cf lsh.jar LSH*.class
	hadoop fs -rm -r -f /user/$(STUDENT_ID)/lsh > /dev/null
	hadoop fs -mkdir /user/$(STUDENT_ID)/lsh > /dev/null
	hadoop fs -put $(INPUT) /user/$(STUDENT_ID)/lsh > /dev/null
	cd lsh_java_files && yarn jar lsh.jar LSH /user/$(STUDENT_ID)/lsh /user/$(STUDENT_ID)/lsh/output || true
	hadoop fs -cat /user/$(STUDENT_ID)/lsh/subtask1/part-r-0* > output_subtask1.txt || true
	hadoop fs -cat /user/$(STUDENT_ID)/lsh/subtask2/part-r-0* > output_subtask2.txt || true
	hadoop fs -cat /user/$(STUDENT_ID)/lsh/output/part-r-0* > output_final.txt || true
	hadoop fs -rm -r -f /user/$(STUDENT_ID)/lsh > /dev/null
all_py:
	hadoop fs -rm -r -f /user/$(STUDENT_ID)/lsh > /dev/null
	hadoop fs -mkdir /user/$(STUDENT_ID)/lsh > /dev/null
	hadoop fs -put $(INPUT) /user/$(STUDENT_ID)/lsh > /dev/null
	cd lsh_py_files && python3 run.py /user/$(STUDENT_ID)/lsh || true
	hadoop fs -cat /user/$(STUDENT_ID)/lsh/subtask1/part-0* > output_subtask1.txt || true
	hadoop fs -cat /user/$(STUDENT_ID)/lsh/subtask2/part-0* > output_subtask2.txt || true
	hadoop fs -cat /user/$(STUDENT_ID)/lsh/output/part-0* > output_final.txt || true
	hadoop fs -rm -r -f /user/$(STUDENT_ID)/lsh > /dev/null

