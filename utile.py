import numpy as np
import os

test_root = './fluccs_data/'
exe_file = 'test.sh'

def test_gene(n):
	vals = np.random.randint(156, size=n)
	trains = []
	for i in range(156):
		if i not in vals:
			trains.append(i)
	return  vals, trains

def cmd_gene(vals,trains):
	cmd_train = 'python3 train.py '
	cmd_val = 'python3 validation.py '
	for root,dirs,files in os.walk(test_root):
		for i in range(1,len(files)):
			if i-1 in vals:
				#print(i-1)
				cmd_val+=root+files[i]+' '
			else:
				cmd_train+=root+files[i]+' '
	return cmd_train,cmd_val

def exe_gene():
	vals, trains = test_gene(10)
	#print(vals)	 
	cmd_train,cmd_val = cmd_gene(vals,trains)
	with open(exe_file,'w') as f:
		f.write('#!/bin/bash\n')
		f.write(cmd_train+'\n')
		f.write(cmd_val+'\n')
	#print(cmd_val)
	return 0

exe_gene()