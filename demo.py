import rcc
import pdb
import numpy as np
from sklearn.metrics import adjusted_mutual_info_score

X = []
Y = []

with open('pendigits.txt', 'r') as f:
	for line in f:
		line_split = line.strip().replace(' ','').split(',')
		x = np.array([int(s) for s in line_split[:-1]])
		y = int(line_split[-1])
		X.append(x)
		Y.append(y)

X = np.array(X).astype(np.float32)
Y = np.array(Y)
clusterer = rcc.rcc_cluster(measure='cosine')
P = clusterer.fit(X)
P = clusterer.labels_
print('AMI: {}'.format(adjusted_mutual_info_score(Y, P)))
