from __future__ import division
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import pickle




# ___________data_utile______________
def loadData(files):
	X = []
	for file in files:
		data = np.genfromtxt(file, delimiter=',')
		M = len(data)-1
		F = len(data[0])-1
		x = np.zeros((M,F),dtype = np.float64)
		for j in range(M):
			for k in range(F):
				x[j][k] = data[j+1][k+1]
		X.append(x)
	return np.array(X)

# return the best feature to use for each file
def bestFeatures(data):
	best_features = []
	for f in data:
		rank_min = 10000
		best_f = -1
		for i in range(41):
			rank = fitnessOne(f,f[:,i])
			if rank < rank_min:
				rank_min = rank
				best_f = i
		best_features.append([best_f,rank_min])
	return np.array(best_features)

# return the useful features based on this dataset
def usefulFeatures(best_f):
	uf  = [0]
	for f in best_f:
		if f[0]!=0:
			uf.append(f[0])
	return list(set(uf))


# ___________analyse_________________
def ana_ranks(ranks):
	top1_num = 0
	top3_num = 0
	top5_num = 0
	for r in ranks:
		if r==0:
			top1_num+=1
		if r<3:
			top3_num+=1
		if r<5:
			top5_num+=1
	return top1_num, top3_num, top5_num

# input one file and score array for this file 
# output the rank of the real method
def fitnessOne(fault,score):
	rank = np.argsort(-score)
	f = np.nonzero(fault[:,-1])
	r = np.argwhere(rank == f[0][0])[0][0]
	while(r!=0 and score[r-1]==score[r]):
		r = r-1
	return r

# output ranks of a group of files
def fitness(data, score):
	N = len(data)
	fs_sum = 0
	ranks=[]
	for d, s in zip(data, score):
		r = fitnessOne(d,s)
		ranks.append(r)
		fs_sum += r
	analyse = ana_ranks(ranks)
	fs = fs_sum / N
	return fs, np.array(ranks)    

def printRanks(ranks):
	for i in ranks:
		print (i)


#_______________fc_mode________________

#feature classification:take each feature as a class
#for each bug,calculate its means and vars for each feature
def fc_standardData(data):
	dist = []
	for f in data:
		var = np.var(f[:,:41],axis=0)
		mean = np.mean(f[:,:41],axis=0)
		maxi = np.max(f[:,:41],axis=0)
		mini = np.min(f[:,:41],axis=0)
		range = maxi-mini
		d = np.array([mean,var,maxi,mini,range])
		dist.append(d)
	dist = np.array(dist)
	return dist

# construct labels for each file ---> y
# to reduce the comlexity of learning, we map each useful feature to its
# index in uf list, in this way the number of the classes is the number of 
# the useful features
def fc_consLabels(data,uf,bf):
	N = len(data)
	l = np.zeros(N)
	for i in range(N):
		if bf[i][0]!=0:
			l[i] = uf.index(bf[i][0])
	return l

# return the label to real labels
def fc_toRealLabel(uf,out):
	l = np.zeros(len(out))
	for i in range(len(out)):
		if out[i]!=0:
			l[i] = uf[int(out[i])]
	return l

def fc_fit(data,out):
	ranks = []
	for i in range(len(out)):
		rank = fitnessOne(data[i],data[i][:,int(out[i])])
		ranks.append(rank)
	acc = np.array(list(ana_ranks(ranks)))/len(ranks)
	return ranks,acc
		
def fc_train(X,y):
	#param_gid={'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000] }
	X = StandardScaler().fit_transform(X)
	#clf_l2_LR_fc = LogisticRegression(C = 1.0,penalty='l2', tol=1e-6)
	clf_l2_LR_fc = LogisticRegression(C = 1.0, tol=1e-6, penalty = 'l2')
	clf_l2_LR_fc.fit(X,y)
	coef_l2_LR_fc = clf_l2_LR_fc.coef_.ravel()
	return clf_l2_LR_fc

def fc_val(X):
	model = pickle.load(open('trained_model', 'rb'))
	y = model.predict(X)
	return y

def train(data,uf):
	#print('training...')
	N = len(data)
	bf = bestFeatures(data)
	X = fc_standardData(data)
	X = X.reshape(N,5*41)
	y = fc_consLabels(data,uf,bf)
	#print(y)
	model = fc_train(X,y)
	pickle.dump(model, open('trained_model', 'wb'))
	return 0

def validation(test_data,uf):
	#print('testing...')
	N = len(test_data)
	X = fc_standardData(test_data)
	X = X.reshape(N,5*41) 
	y = fc_val(X)
	y = fc_toRealLabel(uf,y)
	ranks,acc = fc_fit(test_data,y)
	return ranks,acc

