import numpy as np

def generator(X,y,steps_per_epoch):
    while True:
        for i in range(steps_per_epoch):
            start = (X.shape[0] // steps_per_epoch)*i
            if i == steps_per_epoch - 1:
                end = X.shape[0]
            else:
                end  = (X.shape[0] // steps_per_epoch)*(i+1)
            yield np.squeeze(np.stack((X[start:end],) * 3, -1)), y[start:end]