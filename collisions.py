import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from math import atan, atan2, pi
from matplotlib import style
#style.use('dark_background')


m1 = 100.
m2 = 1.

v1s = -1
v2s = 0

const = m1*v1s + m2*v2s
const_e = 0.5*m1*(v1s**2) + 0.5*m2*(v2s**2)
slope = -(m1/m2)**0.5
#print(const, const_e, slope)
v1=[]
v2=[]

cv1=[]
cv2=[]

v1.append(v1s)
v2.append(v2s)

v1p = ((m1- m2)*v1[0])/(m1 + m2)
v2p = v1p + v1[0]

cv1.append(v1s)
cv2.append(v2s)
cv1.append(v1s)
cv2.append(-v2s)

v1.append(v1p)
v1.append(v1p)
v2.append(v2p)
v2.append(-v2p)

v1p=v1[1]
v2p=v2[1]
print(v1p, v2p)

i=2
#while v1[i] > 0 and v2[i] <0 or v1[i] < 0 and v2[i] > 0:
while v1p > 0 and -v2p < 0 and v1p < 0  and -v2p > 0:
    i += 1
    v2p = (((2 * m1) * v1[i-1])/(m1 + m2)) - (((m1 - m2) * v2[i-1])/(m1 + m2))
    v1p = (((2 * m2) * v2[i-1])/(m1 + m2)) + (((m1 - m2) * v1[i-1])/(m1 + m2)) 
    v1.append(v1p)
    v1.append(v1p)
    v2.append(v2p)
    v2.append(-v2p)
    # half circle
    cv1.append(v1p)
    cv2.append(-v2p)

    i += 1

while v2[i] > v1[i]:
    i += 1
    v2p = (((2 * m1) * v1[i-1])/(m1 + m2)) - (((m1 - m2) * v2[i-1])/(m1 + m2))
    v1p = (((2 * m2) * v2[i-1])/(m1 + m2)) + (((m1 - m2) * v1[i-1])/(m1 + m2)) 
    v1.append(v1p)
    v1.append(v1p)
    v2.append(v2p)
    v2.append(-v2p)
    # half circle
    cv1.append(v1p)
    cv2.append(-v2p)

    i += 1
        
v1round = [round(v, 2) for v in v1]
v2round = [round(v, 2) for v in v2]
#print(v1round, '\n')
#print(v2round)
print(len(v1round) - 2)

nv1 = [v*(m1**0.5) for v in v1] 
nv2 = [v*(m2**0.5) for v in v2]
# circle normalization
ncv1 = [v*(m1**0.5) for v in cv1]
ncv2 = [v*(m2**0.5) for v in cv2]

negcv1 = [-v for v in ncv1]
negcv2 = [-v for v in ncv2]

for i, j in enumerate(negcv1):
    ncv1.append(j)

for i, j in enumerate(negcv2):
    ncv2.append(j)

circle = plt.Circle((0.,0.), const_e, color='green' ,fill=False)
fig, ax = plt.subplots(figsize=(6,6))
ax.plot(nv1, nv2)
ax.scatter(nv1, nv2, color='red')
ax.add_artist(circle)
plt.show()



