import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from math import atan, atan2, pi
from matplotlib import style
style.use('dark_background')


m1 = 10.
m2 = 1.

v1s = -1
v2s = 0

const = m1*v1s + m2*v2s
const_e = 0.5*m1*(v1s**2) + 0.5*m2*(v2s**2)
# v1=v2
v0 = (2*const_e/(m1+m2))**0.5
slope = -(m1/m2)**0.5
#print(const, const_e, slope)
phi = np.linspace(0, 2*np.pi, 100)
r = np.sqrt(2*const_e)

x = r*np.cos(phi)
y = r*np.sin(phi)


v1=[]
v2=[]



v1.append(v1s)
v2.append(v2s)

v1p = ((m1- m2)*v1[0])/(m1 + m2)
v2p = v1p + v1[0]


v1.append(v1p)
v1.append(v1p)
v2.append(v2p)
v2.append(-v2p)

v1p=v1[1]
v2p=v2[1]
print(v1p, v2p)

i=2

while v1p > 0 and -v2p < 0 and v1p < 0  and -v2p > 0:
    i += 1
    v2p = (((2 * m1) * v1[i-1])/(m1 + m2)) - (((m1 - m2) * v2[i-1])/(m1 + m2))
    v1p = (((2 * m2) * v2[i-1])/(m1 + m2)) + (((m1 - m2) * v1[i-1])/(m1 + m2)) 
    v1.append(v1p)
    v1.append(v1p)
    v2.append(v2p)
    v2.append(-v2p)

    i += 1

while v2[i] > v1[i]:
    i += 1
    v2p = (((2 * m1) * v1[i-1])/(m1 + m2)) - (((m1 - m2) * v2[i-1])/(m1 + m2))
    v1p = (((2 * m2) * v2[i-1])/(m1 + m2)) + (((m1 - m2) * v1[i-1])/(m1 + m2)) 
    v1.append(v1p)
    v1.append(v1p)
    v2.append(v2p)
    v2.append(-v2p)

    i += 1
        
v1round = [round(v, 2) for v in v1]
v2round = [round(v, 2) for v in v2]

nv1 = [v*(m1**0.5) for v in v1] 
nv2 = [v*(m2**0.5) for v in v2]

v01 = np.linspace(0, v0*(m1**0.5), len(y))
v02 = np.linspace(0, v0*(m2**0.5), len(y))
y0 = []
for i, j in enumerate(y):
    if 0 < j < v02[-1]:
        if x[i] > 0:
            y0.append(j)

numo = int(len(nv2)*0.1)
nv2r = nv2[-numo:]
o = []
t = 1

for i, j in enumerate(nv2r):
    if 0 < j < v02[-1] and x[i]>0:
        o = t
        break
    t += 1

num = abs(t - numo) - 1
if num == numo:
    num = 0
elif num < 0:
    num = 0
print(t, ' ',num , ' ',numo)

x1 = np.array(nv1)
x2 = np.array(nv2)
y0.reverse()
y0 = np.append(v02, y0)
x0 = np.linspace(0, r, len(y0))

x1round = [round(x, 2) for x in x1]
x2round = [round(x, 2) for x in x2]

num_col = len(x1)
print(f'number of colisions: {num_col}')

fig, ax = plt.subplots(figsize=(8,8))
p1, = ax.plot(x1, x2, animated=True)
p2, = ax.plot(x1, x2, 'o', color='IndianRed',label='collison')
ax.plot(x,y)
ax.fill_between(x0, 0 , y0, facecolor='IndianRed', alpha=0.5)
title = ax.text(.5,.7, "", bbox={'facecolor':'black', 'alpha':0.5, 'pad':5}, transform=plt.gcf().transFigure, ha="center")

def update(num, x1, x2, x1round, x2round, n, p, p1):
    p.set_data(x1[:num], x2[:num])
    p1.set_data(x1[:num], x2[:num])
    title.set_text(f'Number of collisions {len(x1[:num])-n} \n' +'$\mathregular{v_1}$'+ f'={x1round[num]} \n' '$\mathregular{v_2}$' + f'={x2round[num]}')
    return p, p1, title

label4 = '$\mathregular{v_1}$ = $\mathregular{v_2}$'
p4, = ax.plot(v01, v02, color='IndianRed', linestyle='--', label=label4)
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05))
ax.set_aspect(1)

print(len(nv1))
print(len(x1)-num)

ani = animation.FuncAnimation(fig,update, len(nv1), fargs=[x1, x2, x1round, x2round, num, p1, p2], interval=0.0000001, repeat=False, blit=True)

plt.show()





