from __future__ import print_function
import re
import binascii
import sys
from pyspark import SparkContext
import numpy as np
from scipy.spatial import distance

# function to assign every data point to its closest center
def closestPoint(p, centers):
	bestIndex = 0
    	closest = float("+inf")
    	for i in range(len(centers)):
		tempDist = distance.euclidean(p,j[i])
        	if tempDist < closest:
			closest = tempDist
			bestIndex = i
   	return bestIndex

if __name__ == "__main__":
	if len(sys.argv) != 4:
        	print("Usage: spark-submit kmeans.py  <inputfile> <outputfile> <number of clusters,k>", file=sys.stderr)
        	exit(-1)

	sc = SparkContext(appName="K means with minhashing")
	file = sc.wholeTextFiles(sys.argv[1])
	content = file.values().collect()
	# data preprocessing - remove special characters, punctuation, html tags and rogue characters
	for article in content:
		i = article.replace("\r\n\r\n\r\n\r\n\r\n\r\n\r\n\r\n\r\n\r\n\r\n\"",'')

	articles = i.split("\r\n\"")
	l = articles[2640]+articles[2641]
	articles = articles[1:2640]
	articles.append(l)
	
	art = [w.replace('strong>','') for w in articles]
	art = [w.replace('</strong','') for w in art]
	art = [re.sub('img([\w\W]+?)""\/','',w) for w in art]
	art = [re.sub('<[^<]+?>','',w) for w in art]
	art = [re.sub('&amp;','',w) for w in art]
	art = [re.sub('[ ](?=[ ])|[^,.A-Za-z ]+', '', w) for w in art]
	art = [re.sub('[ ](?=[ ])|[^-A-Za-z ]+', ' ', w) for w in art]
	art = [w.lower() for w in art]
	
	#remove stop words
	stop_words = sc.textFile("stopWords.txt").collect()
	art = [i.split(" ") for i in art]
	r = []
	for i in range(len(art)):
		r.append(' '.join(w for w in art[i] if not w in stop_words))
	
	# get unique list of one word shingles for each document
	r = [i.split(" ") for i in r]
	r = [list(set(i)) for i in r]

	words = sc.parallelize(r)
	words = words.zipWithIndex().map(lambda (x,index):(index,x))

	# hash the shingles
	words = words.map(lambda (index,xs):(index,[binascii.crc32(x) & 0xffffffff for x in xs]))
	e=words.flatMap(lambda (index,xs):[(index,x) for x in xs])
	random32bit = sc.textFile("randomInt.txt").map(lambda x:int(x)).collect()
	signatures = sc.parallelize([])

	#generate minhash signatures for all documents by xoring the shingles with a list of random 32 bit integers 
	for i in range(100):
		signature = e.map(lambda (k,r):(k,r^random32bit[i])).reduceByKey(lambda x,y:min(x,y))
		signature = signature.map(lambda (k,r):((k,i),r))
		signatures = signatures.union(signature)
	
	g = signatures.sortByKey().map(lambda ((k,i),r):(k,r)).groupByKey().mapValues(list)

	#take random sample of k data points from our document set as specified by command line argument
	j = g.takeSample(False,int(sys.argv[3]),1)
	j = [x[1] for x in j]
	
	# run kmeans for 20 iterations
	for i in range(20):
		closest = g.map(lambda (k,v): (closestPoint(v,j), (v, 1, k)))
		points = closest.reduceByKey(lambda p1, p2: (np.add(p1[0],p2[0]), p1[1] + p2[1]))
		newPoints = points.map(lambda (k,(u,v)): (k, np.divide(u,v))).collect()
		for (idex, p) in newPoints:
			j[idex] = p
	
	#write clusters to file
	clusters = closest.map(lambda (k,(a,b,c)):(k,c))
	clusters = clusters.groupByKey().mapValues(list)
	clusters.saveAsTextFile(sys.argv[2])
	sc.stop()

