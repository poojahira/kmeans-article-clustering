This project aimed to cluster of news articles using iterative manually implemented Kmeans in PySpark. The dataset contains a list of short news articles in a CSV format, where each row contains the columns – the short article, date of publication, title and category. 

Stop-words, punctuation, special and rogue characters, duplicate words and html tags were removed. One-word shingling was implemented. A minhash signature of length 100 was used. The shingles were hashed and XORed with 100 random 32 bit numbers for minhashing. A random sample of 5 datapoints were chosen as seed and kmeans was run for 20 iterations. 

Run the program as:
spark-submit kmeans.py <inputfile> <outputfile> <number of clusters,k>
