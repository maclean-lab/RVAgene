########### Author:  Raktim Mitra  email: timkartar7879@gmail.com ##################
from rvagene.rvagene import RVAgene
from rvagene.utils import open_data
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d import Axes3D
from torch.utils.data import TensorDataset
import sys


hidden_size = 70
hidden_layer_depth = 1
latent_length = 3
batch_size = 20
learning_rate = 0.003
n_epochs = 400
dropout_rate = 0.2
optimizer = 'Adam' # options: ADAM, SGD
cuda = False # options: True, False
print_every=30
clip = True # options: True, False
max_grad_norm=5
loss = 'MSELoss' # options: SmoothL1Loss, MSELoss
dataset=sys.argv[1]

X_train, _, y_train, _ = open_data('data', ratio_train=1, dataset=dataset)

num_classes = len(np.unique(y_train))
base = np.min(y_train)  # Check if data is 0-based
if base != 0:
    y_train -= base


train_dataset = TensorDataset(torch.from_numpy(X_train))


sequence_length = X_train.shape[1]
number_of_features = X_train.shape[2]

################## define model and train on synthetic data ##########################
rvagene = RVAgene(sequence_length=sequence_length,
            number_of_features = number_of_features,
            hidden_size = hidden_size, 
            hidden_layer_depth = hidden_layer_depth,
            latent_length = latent_length,
            batch_size = batch_size,
            learning_rate = learning_rate,
            n_epochs = n_epochs,
            dropout_rate = dropout_rate,
            optimizer = optimizer, 
            cuda = cuda,
            print_every=print_every, 
            clip=clip, 
            max_grad_norm=max_grad_norm,
            loss = loss)

rvagene.fit(train_dataset)

z_run = rvagene.transform(train_dataset)

############### plot 3 dimensional z plot ##################
fig = plt.figure()
ax1 = fig.add_subplot(111, projection='3d')

ax1.scatter(z_run[:,0],z_run[:,1],z_run[:,2])
plt.savefig("figs/z_run_"+dataset+".png", dpi=500)
plt.close()

################## k means clustering on z ################
from sklearn.cluster import MiniBatchKMeans
kmeans = MiniBatchKMeans(n_clusters=6,
                         random_state=0,
                         batch_size=6,
                         max_iter=10).fit(z_run)

centers = kmeans.cluster_centers_

c_zs = kmeans.predict(z_run)
legend = dict()
hex_colors = []
counts = dict()

################## plot 3 D z with kmeans clustering ##############
for i in np.unique(c_zs):
    hex_colors.append('#%06X' % np.random.randint(0, 0xFFFFFF))
    legend[i]=hex_colors[-1]
colors = [hex_colors[int(i)] for i in c_zs]
fig = plt.figure()
ax1 = fig.add_subplot(111, projection='3d')

ax1.scatter(z_run[:,0],z_run[:,1],z_run[:,2],c=colors)

legend_elements = [Patch(facecolor=legend[i],
                         label="cluster "+str(i) ) for i in legend.keys() ]
ax1.legend(handles=legend_elements)
plt.savefig("figs/clusters_"+dataset+".png", dpi=500)
plt.close()

#################### sample points from Z around each cluster mean detected by K-means, decode and plot those points #################
samples_per_cluster = 20


length = sequence_length
x= np.linspace(0,length/1000,length)

for i in range(centers.shape[0]):
    mean = centers[i]
    cov = np.identity(centers[i].shape[0]).astype(np.double)*0.3
    
    samples = []
    for gene in range(samples_per_cluster):
        sample = np.random.multivariate_normal(mean,cov).astype(np.double)
        samples.append(sample)
        print("sample ",gene,"for cluster ", i,": ",sample)
    torch_samples = torch.from_numpy(np.array(samples)).float()

    mean_to_torch = torch.from_numpy(np.array([mean]*samples_per_cluster)).float()
    if torch.cuda.is_available():  
        dev = "cuda:0" 
    else:  
        dev = "cpu"  
    device = torch.device(dev)  
    torch_samples = torch_samples.to(device)
    mean_to_torch = mean_to_torch.to(device)
    device = torch.device("cpu")
    regen = np.squeeze((rvagene.decoder.forward(torch_samples)).to(device).data.numpy())
    mean_regen = np.squeeze((rvagene.decoder.forward(mean_to_torch)).to(device).data.numpy())
    print("shape of generated cluster:", regen.shape)

    ############### plot decoded cluster means ##############
    for gene in range(samples_per_cluster):    
        plt.plot(x,regen[:,gene],alpha=0.6,linewidth=0.9)
    plt.plot(x,mean_regen[:,0],color="#000000")
    plt.legend(handles=[legend_elements[i]])
    plt.savefig("figs/generated_cluster_" + dataset + str(i) + ".png", dpi=500)
    plt.close()


