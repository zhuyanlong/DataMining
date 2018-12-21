import numpy as np

def MinkowskDis(v1,v2,h):
	dis=np.sum(np.power(np.power(np.fabs(v1-v2),h),1/h))
	print(dis)

def main():
	d1=[1,2,3,4,5,6]
	d2=[2,3,4,5,6,7]
	v1=np.array(d1)
	v2=np.array(d2)
	MinkowskDis(v1,v2,2)

main()