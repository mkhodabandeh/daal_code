import sys, os
sys.path.append(os.getcwd())
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pylab import *
from matplotlib.widgets import Slider

def get_gan_data1(num=1000):
    print num
    d1 = np.random.multivariate_normal([0,0], cov=[[1,0.01],[0.01,1]], size=2*num)
    l1 = np.ones((len(d1),))*1
    return d1, l1 
def get_gan_data(num=1000):
    print num
    d1 = np.random.multivariate_normal([0,0], cov=[[300,350],[250,200]], size=num)
    d2 = np.random.multivariate_normal([50,50], cov=[[200,-450],[-250,150]], size=num)
    l1 = np.ones((len(d1),))*1
    l2 = np.ones((len(d2),))*2
    data = np.concatenate((d1,d2), axis=0)
    print 'gen data', data.shape
    label = np.concatenate((l1,l2), axis=0) 
    return data, label

def get_data(num=300):
    d1 = np.random.multivariate_normal([0,0], cov=[[300,350],[250,200]], size=num)
    d2 = np.random.multivariate_normal([50,50], cov=[[200,-450],[-250,150]], size=num)
    d3 = np.random.multivariate_normal([-15,50], cov=[[150,20],[20,150]], size=int(num/6))
    d4 = np.random.uniform(-80, 130, size=(int(num/3),2))
    l1 = np.ones((len(d1),))*1
    l2 = np.ones((len(d2),))*2
    l3 = np.ones((len(d3),))*3
    l4 = np.ones((len(d4),))*4
    plt.scatter(d1[:,0], d1[:,1], marker='o',)
    plt.scatter(d2[:,0], d2[:,1], marker='x' )
    plt.scatter(d3[:,0], d3[:,1], marker='^' )
    plt.scatter(d4[:,0], d4[:,1], marker='<' )
    plt.show()
    plt.savefig('fig.png')
    data = np.concatenate((d1,d2,d3,d4), axis=0)
    label = np.concatenate((l1,l2,l3,l4), axis=0) 
    return data, label


def load_data():
    try:
        data = np.load('data.npy')
        label= np.load('label.npy')
        gan_data = np.load('gan_data.npy')
        gan_label= np.load('gan_label.npy')
	print 'try'       
    except:
        data,label = get_data()
        np.save('data.npy', data)
        np.save('label.npy', label)
        #gan_data,gan_label = get_gan_data()
        gan_data,gan_label = get_gan_data1()
        np.save('gan_data.npy', gan_data)
        np.save('gan_label.npy', gan_label)
	print 'except'
    print 'gan_data',gan_data.shape
    print 'data',data.shape
    return (data, label), (gan_data, gan_label) 
#get_data()

def train_gan():
    pass

def visualize(data, label, al_scores, dx_scores):
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    def updatef(beta):
        new_c = al_scores * (dx_scores ** beta)
        ax.clear()
        scatter = ax.scatter(data[:,0], data[:,1], c=new_c)
        draw()

    scatter = ax.scatter(data[:,0], data[:,1], c=al_scores)
    plt.subplots_adjust(left=0.25, bottom=0.25)
    axcolor = 'lightgoldenrodyellow'
    ax_beta = plt.axes([0.25, 0.15, 0.65, 0.03], axisbg=axcolor)
    beta_slider= Slider(ax_beta, 'beta', 0.0, 2.0, valinit=0)
    beta_slider.on_changed(updatef)
    plt.savefig('fig.png')
    show()
    #for i in set(label.tolist()):
    #    plt.scatter(data[:,0], data[:,1],
def normalize(arr):
    arr = arr.copy() - np.min(arr)
    arr = arr / np.max(arr)
    return arr

def test():
    #(data, label), (gan_data, gan_label) = load_data()
    #fig = plt.figure()
    #ax = fig.add_subplot(111)
    #scatter = ax.scatter(gan_data[:,0], gan_data[:,1])
    #fig.savefig('gan_data.png')
    #dist2 = data - np.zeros((len(data), 2))
    #dist2 = dist2*dist2
    #dist2 = np.sqrt(dist2[:,0]+dist2[:,1])
    #dist1 = data - np.ones((len(data), 2))*50
    #dist1 = dist1*dist1
    #dist1 = np.sqrt(dist1[:,0]+dist1[:,1])
    data = np.load('X.npy')
    label = np.ones((len(data),))
    dist1 = np.load('ent.npy')
    dist1 = normalize(dist1)
    dist2 = np.load('Px.npy')
    visualize(data, label, dist1, dist2) 




if __name__=='__main__':
    test()
    #pass
