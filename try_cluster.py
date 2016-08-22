
import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn import metrics
import pylab

filename_out = './data/train_short.csv'

filename_in = './data/train.csv'
my_data = pd.read_csv(filename_in,nrows=5000);
my_data.to_csv(filename_out,index = False);

my_data = pd.read_csv(filename_out);



#my_data['Sequence'] = my_data['Sequence'].apply(str.split,args=',')

x = map(lambda x: [int(i) for i in x], my_data['Sequence'].apply(str.split,args=',').values)
seq_lenght = 15

x = [i[:seq_lenght] for i in x if len(i)>seq_lenght]
print len(x)

model = TSNE(n_components=2, random_state=0)

X = np.array(model.fit_transform(np.array(x)))



## ===============
## 
## ===============
###############################################################################
# Compute DBSCAN
db = DBSCAN(eps=1,min_samples=7).fit_predict(X)
#eps=0.3, min_samples=3
print db
# db = [np.abs(i+0.1)/np.max(db) for i in db]
Xsubset = [X[i] for i in range(len(X)) if db[i]==16]

plt.figure(2);
plt.scatter(X[:,0],X[:,1], s=75,c=db, cmap=pylab.cm.jet)
plt.colorbar()

plt.figure(2);
plt.scatter(Xsubset[:,0],Xsubset[:,1])

plt.show()



