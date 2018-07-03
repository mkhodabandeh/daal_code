import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json

def errorfill(x, y, yerr, color=None, alpha_fill=0.3, ax=None, xticks=None):
    ax = ax if ax is not None else plt.gca()
    if color is None:
        color = ax._get_lines.color_cycle.next()
    if np.isscalar(yerr) or len(yerr) == len(y):
        ymin = y - yerr
        ymax = y + yerr
    elif len(yerr) == 2:
        ymin, ymax = yerr
    h = ax.plot(x, y, color=color)[0]
    if xticks is None:
	plt.xticks(x, map(str, x))
    else:
        plt.xticks(xticks)
    ax.fill_between(x, ymax, ymin, color=color, alpha=alpha_fill)
    return h

def to_np2d(arr):
    arr = np.expand_dims(np.array(arr).astype(float),0)
    return arr 

if 1 == 2:
	beta_0_accs = {}
	for i in range(1):
	   print '#### {} ####'.format(i)
	   flag = 0
	   p = 'beta_size2acc_new/all_size2acc.json'.format(i)
	   try:
	       with open(p, 'r') as f:
		  jj= json.load(f)
	   except:
	      continue
	   for dd in jj:
	       for key in dd.keys():
		   val = dd[key]
		   if key in beta_0_accs:
		      if type(val) == list:
			  beta_0_accs[key].extend(val)
		   else:
		      if type(val) == list:
			  beta_0_accs[key] = val
		      else:
			  beta_0_accs[key] = []
	"""" ours """
	sorted_accs = []
	for key,val in beta_0_accs.iteritems():
	    md = np.mean(val)
	    std = np.std(val)
	    sorted_accs.append((float(str(key)), (md,std)))
	print sorted_accs[0]
	sorted_accs.sort(key=lambda x: x[0])
	accs = [key for key,val in sorted_accs]
	md = [val[0] for key,val in sorted_accs]
	std = [val[1] for key,val in sorted_accs]
	print md, std
	#plt.figure()
	with open('beta_size2acc.json', 'w') as f:
	    json.dump(
		{
		'accs':accs,
		'md':md,
		'std':std,
		}, f)


plt.figure()
with open('beta_size2acc.json', 'r') as f:
  jj= json.load(f)
std1 = jj['std']
md1 = jj['md']
accs1 = jj['accs']
with open('outlier_size2acc.json', 'r') as f:
  jj= json.load(f)
std = jj['std']
md = jj['md']
accs = jj['accs']
#xticks = np.arange(35,91, step=5).tolist()+[np.max(accs), np.max(accs1)]
xticks = np.arange(35,96, step=5)
xticks.sort()
#plt.axvline(x=np.max(accs), linestyle='dashed', color='lightgrey')
#plt.axvline(x=95, linestyle='dashed', color='lightgrey')
plt.axvline(x=np.max(accs1), linestyle='dashed', color='lightgrey')
plt.axhline(y=np.max(md1), linestyle='dashed', color='lightgrey')
print np.max(md1)
h1=errorfill(np.array(accs1), np.array(md1), np.array(std1), color='dodgerblue', xticks=np.arange(0,100, step=5))
h=errorfill(np.array(accs), np.array(md), np.array(std), color='r', xticks=np.array(xticks))
plt.xticks(np.array(xticks), rotation=45,fontsize=6) 
yticks = np.arange(0, 6001, step=1000).tolist()+[500]
yticks.sort()
plt.yticks(np.array(yticks)) 
plt.legend(handles=[h1,h], labels=['Ours', 'Baseline'])
plt.ylabel('Number of queries')
plt.xlabel('Accuracy (%)')
plt.savefig('fig6c_final.pdf')
