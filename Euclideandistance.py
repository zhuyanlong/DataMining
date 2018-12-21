import numpy as np

def EuclidDis(v1,v2):
	distance=np.sqrt(np.sum(np.square(v1-v2)))
	print(distance)


def main():
	d1=[1,2,3,4,5,6]
	d2=[2,3,4,5,6,7]
	v1=np.array(d1)
	v2=np.array(d2)
	EuclidDis(v1,v2)

main()