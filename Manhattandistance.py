import numpy as np

def ManhattanDis(v1,v2):
	dis=np.sum(np.fabs(v1-v2))
	print(dis)

def main():
	d1=[1,-4,3,6,5,6]
	d2=[2,3,4,5,6,7]
	v1=np.array(d1)
	v2=np.array(d2)
	ManhattanDis(v1,v2)

main()