import numpy as np

def CosineSim(v1,v2):
	value_v1=np.linalg.norm(v1,2)
	value_v2=np.linalg.norm(v2,2)
	dis=np.dot(v1,v2)/(value_v1*value_v2)
	print(dis)

def main():
	d1=[5,0,3,0,2,0,0,2,0,0]
	d2=[3,0,2,0,1,1,0,1,1,0]
	v1=np.array(d1)
	v2=np.array(d2)
	CosineSim(v1,v2)
	

main()