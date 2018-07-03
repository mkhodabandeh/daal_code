import matplotlib.pyplot as plt
import numpy as np
import sys

try:
    filename1 = sys.argv[1]
    filename2 = sys.argv[2]
except:
    print 'WARNING: no argument'
    filename1 = 'npys/accuracies_beta=0.0.npy'
    filename2 = 'npys/accuracies_beta=0.2.npy'
accs_ = np.load(filename1)
accs1 = accs_.copy()
accs2 = np.load(filename2)
accs1[0] = accs2[0]
for i in [1,2,3,4]:
    accs1[i] = accs_[i]-accs_[i-1]+accs1[i-1]
fig, ax = plt.subplots()
ax.plot(range(len(accs1)), accs1, 'xb-', label='Entropy')
ax.plot(range(len(accs2)), accs2, 'xg-', label='Ours')
ax.set_xticks(np.arange(len(accs1)))
ax.set_xticklabels((str(i) for i in range(len(accs1))))
ax.set_ylabel('Accuracy (%)')
ax.legend()
plt.savefig('accs.png')



plt.figure()
men_means= np.load('npys/num_noises_beta=0.0.npy')
women_means= np.load('npys/num_noises_beta=0.2.npy')
ind = np.arange(len(men_means))  # the x locations for the groups
width = 0.35  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(ind - width/2, men_means, width, 
                        color='IndianRed', label='Entropy')
rects2 = ax.bar(ind + width/2, women_means, width, 
                        color='SkyBlue', label='Ours')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Number of outliers')
ax.set_title('Number of outlier queries per each iteration')
ax.set_xticks(ind)
ax.set_xticklabels((str(i) for i in range(len(men_means))))
ax.legend()
plt.savefig('outliers.png')
