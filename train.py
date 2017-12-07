import sys
from fc_model import loadData, train, bestFeatures, usefulFeatures

files = sys.argv[1:]

uf = [0, 3, 4, 37, 7, 40, 39, 10, 12]
data =  loadData(files)
uf2 = usefulFeatures(bestFeatures(data))
uf = list(set(uf+uf2))
train(data,uf)

def toString (l):
	s = 'uf = ['
	for e in l:
		s += str(e)+', ' 
	s = s[:-2]
	s+=']'
	return s

with open('validation.py','w') as f:
	s1 = 'import sys\nfrom fc_model import loadData, validation, printRanks\nfiles = sys.argv[1:]\n'
	s2 = toString(uf)+'\n'
	s3 = 'test_data =  loadData(files)\nranks, acc = validation(test_data,uf)\nprintRanks(ranks)\n'
	f.write(s1)
	f.write(s2)
	f.write(s3)