import numpy as np
from fc_model import loadData, validation,train
from utile import exe_gene
import os
import matplotlib.pyplot as plt


test_root = './fluccs_data/'

l = np.arange(156.0)
#s = np.random.randint(1,155,9)
s = [15,30,45,60,76,92,108,124,140]
vals_list= np.split(l,s)

def list_gene(vals):
	trains = []
	for i in range(156):
		if i not in vals:
			trains.append(i)
	return  vals, trains

def tests_gene(vals,trains):
	v_files = []
	t_files = []
	for root,dirs,files in os.walk(test_root):
		for i in range(1,len(files)):
			if i in vals:
				v_files.append(test_root+files[i])
			else:
				t_files.append(test_root+files[i])
	return v_files, t_files


def cross_val():
	accs = []
	for v in vals_list:
		vals,trains = list_gene(v)
		v_files, t_files = tests_gene(vals,trains)
		train_data = loadData(t_files)
		val_data = loadData(v_files)
		train(train_data)
		ranks,acc = validation(val_data)
		print(acc)
		accs.append(acc)
	return accs

accs = cross_val()
print(accs)
plt.boxplot(accs)
