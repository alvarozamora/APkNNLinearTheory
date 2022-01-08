import numpy as np
from time import time



def equalsky(data, query, r, ids, k=8, bs=1, verb=True):

	if type(bs) != float:
		bs = np.array(bs).reshape(1,1,3)

	if verb:
		print("Computing EqualSky XYZD")
		start = time()

	D = np.abs(data[ids]-query[:,None])
	D = np.minimum(D, bs-D)
	
	#which = np.argmax(np.abs(D),axis=2)
	cr = np.sqrt(D[:,:,0]**2 + D[:,:,1]**2)
    # This selects the points that have a cylindrical radius (perpendicular distance) less than threshold (in the cone, not band)
	which = cr/np.abs(D[:,:,2]) < np.tan(np.pi/3)

	
	box1 = [np.sort(r[i][w==True ],axis=-1)[:k] for i, w in enumerate(which)] #cone 
	box2 = [np.sort(r[i][w==False],axis=-1)[:k] for i, w in enumerate(which)] #band

	Box1 = []
	Box2 = []

	off = 0
	for box in box1:
		if len(box) != k:
			off += 1
		else:
			Box1.append(box)

	off2 = 0
	for box in box2:
		if len(box) != k:
			off2 += 1
		else:
			Box2.append(box)

	cutoff = np.min([len(Box1), len(Box2)])
	Box1 = np.array(Box1)[:cutoff]
	Box2 = np.array(Box2)[:cutoff]

	print(f"There are {off} off entries.")
	print(f"There are {off2} off2 entries.")
	assert cutoff > 0.9*D.shape[0], "Not enough neighbors in band"
	assert Box1.shape[1] == Box2.shape[1] and Box1.shape[1] == k, "not right number of Nearest Neighbors"


	#box1 = np.sort(np.array([np.sort(r[i][w==True],axis=-1)[:k] for i, w in enumerate(which)]),0)  # Line of Sight
	#box2 = np.sort(np.array([np.sort(r[i][w==False],axis=-1)[:k] for i, w in enumerate(which)]),0) # Perpendicular

	#assert (box1.shape == (D.shape[0], k)) & (box2.shape == (D.shape[0], k)), 'Not enough neighbors in band'
	#if np.random.uniform() > 0.99:
	#	print(box1.shape, box2.shape)

	#import pdb; pdb.set_trace()
	#p = (x + y)/2

	#p = np.sort(np.concatenate((x,y),0),0)
	#return box2, box1
	#print(Box2[:,0].mean(), Box1[:,0].mean())
	return Box2, Box1