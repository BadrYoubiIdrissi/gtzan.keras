import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

fig = plt.figure()

X = np.load("data.npy")

def f(i):
    return X[i,:,:,0]

i = 0
im = plt.imshow(f(i), animated=True)

def updatefig(*args):
    global i
    i+=1
    im.set_array(f(i))
    return im,

ani = animation.FuncAnimation(fig, updatefig, interval=100, blit=True)
plt.show()

