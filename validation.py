import sys
from fc_model import loadData, validation, printRanks
files = sys.argv[1:]
uf = [0, 3, 4, 37, 7, 40, 39, 10, 12]
test_data =  loadData(files)
ranks, acc = validation(test_data,uf)
printRanks(ranks)
