import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json

def errorfill(x, y, yerr, color=None, alpha_fill=0.3, ax=None):
    ax = ax if ax is not None else plt.gca()
    if color is None:
        color = ax._get_lines.color_cycle.next()
    if np.isscalar(yerr) or len(yerr) == len(y):
        ymin = y - yerr
        ymax = y + yerr
    elif len(yerr) == 2:
        ymin, ymax = yerr
    ax.plot(x, y, color=color)
    plt.xticks(x, map(str, x))
    ax.fill_between(x, ymax, ymin, color=color, alpha=alpha_fill)

def to_np2d(arr):
    arr = np.expand_dims(np.array(arr).astype(float),0)
    return arr 

def merge(target, aa, func=np.add):
    for key,val in aa.iteritems():
       if key in target:
           #target[key] = func(target[key], np.array(val))
           target[key] = func(target[key], to_np2d(val))
       else:
           target[key] = to_np2d(val)

paths = {
        'accuracy':'/home/mkhodaba/daal_jobs/mnist/exp2/mnist_daal/run={}/accuracies.json',
        'num outliers': '/home/mkhodaba/daal_jobs/mnist/exp2/mnist_daal/run={}/num_outliers.json'
        }
merge_funcs = {
        'accuracy': lambda x,y: np.concatenate((x,y), axis=0),
        'num outliers':  lambda x,y: np.concatenate((x,y), axis=0)
        }
res = { key:{} for key in paths.keys() }
total = 0.0
for i in range(100):
   print '#### {} ####'.format(i)
   flag = 0
   for key,val in paths.iteritems():
        p = val.format(i)
        try:
            with open(p, 'r') as f:
               jj= json.load(f)
            merge(res[key], jj, merge_funcs[key])
            flag = 1
        except:
            pass
   total+=flag
print 'total', total
colors = ['red', 'blue']
for key,val in res.iteritems():
    plt.figure()
    plt.title(key)
    print '*** {} ***'.format(key)
    i = 0
    for k,v in val.iteritems():
        md = np.mean(v, axis=0)
        print '\tMean[beta={}]: {}'.format(k, str(md))
        std = np.std(v, axis=0)
        print '\tstd[beta={}]: {}'.format(k, str(std))
        #plt.errorbar(range(v.shape[1]), md, yerr=std, fmt='--o', capthick=3, elinewidth=0.5)
       
        errorfill(range(v.shape[1]), md, std, color=colors[i])
        i = 1-i
    plt.savefig(key+'.pdf')
