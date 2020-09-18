########### Author:  Raktim Mitra  email: timkartar7879@gmail.com ##################
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from numpy.random import randint
import sys
import os
import warnings
warnings.filterwarnings('ignore')

method = "convolve"
dataset = sys.argv[1]

nclusters = 6
ngenes = [20]*nclusters
length = 100
x= np.linspace(0,length/1000,length)
data = []
for i in range(nclusters):
    if(method == "add"):
        y = randint(0,100)*np.sin(randint(0,100)*x)+randint(0,100)*np.cos(randint(0,100)*x)
        y = MinMaxScaler(feature_range=(-1,1)).fit_transform(y.reshape(-1, 1)).reshape(-1)
    elif(method == "convolve"):
        y = np.convolve(randint(0,100)*np.sin(randint(0,100)*x), randint(0,100)*np.cos(randint(0,100)*x), 'same')
        y = MinMaxScaler(feature_range=(-1,1)).fit_transform(y.reshape(-1, 1)).reshape(-1)
    else:
        print("not implemented")
        sys.exit()
    y_vec = y[:,np.newaxis] + 0.1
    cov = np.dot(y_vec,y_vec.T)/np.dot(y.T,y)
    del(y_vec)
    for m in range(cov.shape[0]):
        for n in range(cov.shape[1]):
            if(abs(m-n) > 50):
                cov[m][n] = 0

    for gene in range(ngenes[i]):
        exp = np.random.multivariate_normal(y,cov)
        #exp = exp + np.random.normal(0,0.1,y.shape[0])
        data.append([str(i)] + [str(j) for j in exp.tolist()])
        plt.plot(x,exp,alpha=0.8,linewidth=0.9)
    plt.plot(x,y, color="#000000", label="cluster mean")
    plt.savefig("./figs/cluster_" + dataset + str(i) + ".png", dpi = 500)
    plt.close()

if(not os.path.exists("./data/"+dataset)):
    os.mkdir("./data/"+dataset)
f = open("./data/"+dataset+"/"+dataset+"_TRAIN","w")
f.write("\n".join([",".join(row) for row in data]))
f.close()

f = open("./data/"+dataset+"/"+dataset+"_TEST","w")
f.write("\n".join([",".join(row) for row in data[:2]]))
f.close()

print("generated synthetic data at ./data/" + dataset)
print("true clusters of generated data plotted at ./figs/" + dataset)
