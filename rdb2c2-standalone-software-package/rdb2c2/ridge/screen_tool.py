# -*- coding: utf-8 -*-
##主要是在算ridge相关的东西
##学习redge
##
import matplotlib
matplotlib.use('Agg')
from matplotlib.pyplot import *
import matplotlib.pyplot as plt
import matplotlib.patches as patches
plt.rc('axes', prop_cycle=(cycler('color', ['#16a085', '#2980b9','#c0392b','#7f8c8d', '#8e44ad','#2ecc71','#2c3e50','#d35400','#bdc3c7',"#f39c12"])))


def checkandmake(x):
	import os
	if not os.path.exists(x):
		os.mkdir(x)


def packzip(*x):
	import types
	import itertools
	temp=tuple([x[0]]+[itertools.repeat(x[i]) for i in range(1,len(x))])
	return zip(*temp)


def build_remove_diag(matrix,diag_remove):
	from numpy import array,zeros,tril_indices,triu_indices,allclose
	mmm=zeros(array(matrix.shape)-diag_remove-1,dtype=matrix.dtype)
	mmm[tril_indices(mmm.shape[0])]=matrix[tril_indices(matrix.shape[0],-diag_remove-1)]
	mmm[triu_indices(mmm.shape[0],1)]=matrix[triu_indices(matrix.shape[0],diag_remove+2)]
	if not allclose(mmm.T,mmm):
		raise ValueError('Wrong matrix')
	return mmm


def calculate(mm,*args, **kwargs):

	#import mbio
	#from mbio import printInfo, printUpdateInfo

	from numpy import allclose, zeros, empty, array, ones, stack
	from numpy import zeros_like, transpose, triu_indices, ones_like
	from numpy import argmax, indices, unique, vstack, save, hstack
	from numpy import newaxis, pi, inf, arccos,isfinite

	from numpy.linalg import inv, norm

	from scipy import ndimage

	import gc

	method,=args
	tag=kwargs.get('tag','')
	#num=kwargs.get('num','')

	if method not in ['ML','AL','NL']:
		raise ValueError('method must be one of ML, AL or NL. got {} instead'.format(method))

	sigmas=arange(1,3.1,.1)
	searchsize=2

	# Build the matrix with diagnal removed
	oldm=mm.copy()
	mm=build_remove_diag(mm,0)

	# Build empty matrixs
	sigmal=len(sigmas)
	shape=list(mm.shape)
	m=zeros([sigmal]+list(shape))
	newshape=list(m.shape)
	infos=empty(newshape+[8],dtype=float)
	lamda=empty(newshape+[2],dtype=float)
	eigv=empty(newshape+[2,2],dtype=float)
	topoint=empty(newshape+[2,2],dtype=float)

	# Build Convolve matrixs
	index=array(ones((searchsize*2+1,searchsize*2+1)).nonzero())
	tempinfo=empty((index.shape[1],6),dtype=float)
	tempinfo[:,0]=1.
	tempinfo[:,1]=index[0]-(searchsize)
	tempinfo[:,2]=index[1]-(searchsize)
	tempinfo[:,3]=0.5*tempinfo[:,1]*tempinfo[:,1]
	tempinfo[:,4]=0.5*tempinfo[:,2]*tempinfo[:,2]
	tempinfo[:,5]=tempinfo[:,1]*tempinfo[:,2]
	tempinfo=((inv(tempinfo.T.dot(tempinfo)).dot(tempinfo.T)))
	tempinfo.resize(6,searchsize*2+1,searchsize*2+1)
	# Convolve need the filter to be inversed
	tempinfo=tempinfo[:,::-1,::-1]

	# Calculate Convolved matrixs
	for i,s in enumerate(sigmas):
		m[i]=ndimage.gaussian_filter(mm,s,mode='mirror')
		for j in range(1,6):
			infos[i,:,:,j-1]=ndimage.convolve(m[i],tempinfo[j],mode='nearest')+1e-100
		infos[i,:,:,5]=infos[i,:,:,2]+infos[i,:,:,3] # xx+yy
		infos[i,:,:,6]=infos[i,:,:,2]-infos[i,:,:,3] # xx-yy
		infos[i,:,:,7]=2*infos[i,:,:,4]              # 2xy
		lamda[i,:,:,0]=(infos[i,:,:,5]-(infos[i,:,:,6]**2+infos[i,:,:,7]**2)**.5)/2
		lamda[i,:,:,1]=(infos[i,:,:,5]+(infos[i,:,:,6]**2+infos[i,:,:,7]**2)**.5)/2
		temp=((lamda[i,:,:]-infos[i,:,:,3:4])/infos[i,:,:,4:5])
		eigv[i,:,:,0,:]=temp/(temp**2+1)**.5
		eigv[i,:,:,1,:]=1./(temp**2+1)**.5
		A=eigv[i,:,:,:,:1]*lamda[i,:,:,0][:,:,newaxis,newaxis]
		A.resize(shape+[2])
		b=-(eigv[i,:,:,:,0]*infos[i,:,:,:2]).sum(2)
		topoint[i,:,:,1,0]=A[:,:,1]
		topoint[i,:,:,1,1]=-A[:,:,0]
		tonorm=norm(topoint[i,:,:,1],axis=2,keepdims=True)
		tonorm[tonorm==0]=1
		topoint[i,:,:,1]/=tonorm
		temp=stack((A,topoint[i,:,:,1]),2)
		temp[norm(topoint[i,:,:,1],axis=2)!=0]=inv(temp[norm(topoint[i,:,:,1],axis=2)!=0])
		temp1=stack((b,zeros_like(b)),2)
		temp1.resize(list(temp1.shape)+[1])
		topoint[i,:,:,0]=(temp*transpose(temp1,(0,1,3,2))).sum(3)
		topoint[i,norm(topoint[i,:,:,1],axis=2)==0,0]=inf

	topoint[topoint[:,:,:,1,1]<0,1]*=-1

	# Basic filter: close to ridge and is ridge
	closeok=((topoint[:,:,:,0]**2).sum(3)<2)*(lamda[:,:,:,0]<0)
	# Just keep the lower half
	# closeok[:,triu_indices(shape[0],1)[0],triu_indices(shape[0],1)[1]]=False

	mask=ones_like(mm)
	masks=[]
	results=[]
	if method=='ML':
		errors= -lamda[[slice(None,None)]+list(mask.nonzero())+[slice(None,None)]].std(axis=(1,2),ddof=1)/(pi**.5)
		strength = (-lamda[:, :, :, 0]).clip(min=0)
		strength[closeok==False]=0
		strength = strength * (sigmas[:, newaxis, newaxis]**1.5)
	elif method=='NL':
		errors=infos[tuple([slice(None,None)]+list(mask.nonzero())+[slice(5,8)])].std(axis=1,ddof=1)**2
		strength=(infos[:,:,:,5:8]**2).clip(min=0)
		strength[closeok==False]=0
		strength=strength[:,:,:,0]*(strength[:,:,:,1]+strength[:,:,:,2])*(sigmas[:,newaxis,newaxis]**6.)
	elif method=='AL':
		errors=lamda[[slice(None,None)]+list(mask.nonzero())+[slice(None,None)]].std(axis=(1,2),ddof=1)
		strength = ((lamda[:, :, :, 0] - lamda[:, :, :, 1])** 2).clip(min=0)
		strength[closeok==False]=0
		strength = strength * (sigmas[:, newaxis, newaxis]**3.)
	else:
		raise ValueError('method is {}?'.format(method))
	argm=argmax(strength,axis=0)
	xind,yind=indices(argm.shape)
	width=sigmas[argm]
	deg=topoint[argm,xind,yind,1].copy()
	deg[deg[:,:,1]<0]*=-1
	deg=arccos(deg[:,:,0].clip(min=-1,max=1))/pi*180
	dist=cross(topoint[argm,xind,yind,0],topoint[argm,xind,yind,1])
	if method=='ML':
		height=(strength[argm,xind,yind]*(2**1.5)*width**.5)
	elif method=='NL':
		height=(strength[argm,xind,yind]*64*width**2)**.25
	elif method=='AL':
		height=(strength[argm,xind,yind]*8*width)**.5
	else:
		raise ValueError('method is {}?'.format(method))

	if method=='ML':
		errors= -lamda[[slice(None,None)]+list(mask.nonzero())+[slice(None,None)]].std(axis=(1,2),ddof=1)/(pi**.5)
		strength = (-(lamda[:, :, :, 0] - errors[:, newaxis, newaxis])).clip(min=0)
		strength[closeok==False]=0
		strength = strength * (sigmas[:, newaxis, newaxis]**1.5)
	elif method=='NL':
		errors=infos[tuple([slice(None,None)]+list(mask.nonzero())+[slice(5,8)])].std(axis=1,ddof=1)**2
		strength=(infos[:,:,:,5:8]**2-errors[:,newaxis,newaxis,:]).clip(min=0)
		strength[closeok==False]=0
		strength=strength[:,:,:,0]*(strength[:,:,:,1]+strength[:,:,:,2])*(sigmas[:,newaxis,newaxis]**6.)
	elif method=='AL':
		errors=lamda[[slice(None,None)]+list(mask.nonzero())+[slice(None,None)]].std(axis=(1,2),ddof=1)
		strength = ((lamda[:, :, :, 0] - lamda[:, :, :, 1])** 2 - 2 * errors[:, newaxis, newaxis]**2).clip(min=0)
		strength[closeok==False]=0
		strength = strength * (sigmas[:, newaxis, newaxis]**3.)
	else:
		raise ValueError('method is {}?'.format(method))

	argm=argmax(strength,axis=0)
	xind,yind=indices(argm.shape)
	dewidth=sigmas[argm]
	dedeg=topoint[argm,xind,yind,1].copy()
	dedeg[dedeg[:,:,1]<0]*=-1
	dedeg=arccos(dedeg[:,:,0].clip(min=-1,max=1))/pi*180
	dedist=cross(topoint[argm,xind,yind,0],topoint[argm,xind,yind,1])
	if method=='ML':
		deheight=(strength[argm,xind,yind]*(2**1.5)*dewidth**.5)
	elif method=='NL':
		deheight=(strength[argm,xind,yind]*64*dewidth**2)**.25
	elif method=='AL':
		deheight=(strength[argm,xind,yind]*8*dewidth)**.5
	else:
		raise ValueError('method is {}?'.format(method))
	temp=stack((height,width,dist,deg,deheight,dewidth,dedist,dedeg),axis=-1) ## 高度 宽度 距离 方向 去噪版本的高度、宽度、距离、方向##
	temp[temp[:,:,0]==0,1]=0
	temp[temp[:,:,0]==0,2]=len(mm)*2
	temp[temp[:,:,0]==0,3]=0
	temp[temp[:,:,4]==0,5]=0
	temp[temp[:,:,4]==0,6]=len(mm)*2
	temp[temp[:,:,4]==0,7]=0

	finaltemp=zeros(list(oldm.shape)+[temp.shape[2]])
	finaltemp[tril_indices(finaltemp.shape[0],-1)]=temp[tril_indices(temp.shape[0],0)]
	finaltemp[triu_indices(finaltemp.shape[0], 1)]=temp[triu_indices(temp.shape[0],0)]

	return finaltemp


from numpy import *
def dealwith(x):
	#from mbio import printInfo
	from numpy import zeros_like,ones_like,save,array,tril_indices,tril_indices_from
	import os

	proname,resnum,seq,deepcnf,seqnum,matrixs,method=x

	#smethod,=x[1:]
	if not len(set([len(seq),len(deepcnf)]+[i.shape[0] for i in matrixs if i is not None]+[i.shape[1] for i in matrixs if i is not None]))==1:
		print "Data not consistant for",num

	resmap="ACDEFGHIKLMNPQRSTVWY"
	resmap={resmap[i]:i for i in range(len(resmap))}
	resmap['X']=slice(None,None)
	## feature not used
	parallelmatrix=array([[  0.77,  1.84,  0.75,  0.67,  2.89,  1.38,  1.08,  5.01,  0.48,  2.64,  1.94,  0.83,  0.50,  0.75,  1.17,  1.11,  2.10,  5.35,  2.32,  2.44],
	[  1.84,  2.52,  1.56,  0.83,  3.62,  1.96,  4.71,  4.65,  0.66,  2.45,  2.68,  0.88,  0.71,  1.00,  0.53,  1.54,  1.63,  5.30,  6.10,  4.59],
	[  0.75,  1.56,  0.43,  0.57,  0.82,  0.90,  1.71,  1.22,  1.21,  0.92,  0.60,  1.22,  0.33,  0.46,  1.83,  1.26,  1.36,  1.62,  0.94,  0.82],
	[  0.67,  0.83,  0.57,  0.04,  1.05,  0.54,  1.22,  1.30,  1.25,  0.83,  0.74,  0.49,  0.20,  0.59,  1.93,  0.73,  0.98,  1.55,  0.89,  1.44],
	[  2.89,  3.62,  0.82,  1.05,  2.77,  2.34,  2.10,  7.09,  0.88,  3.10,  2.93,  1.28,  0.43,  1.24,  1.02,  1.32,  2.16,  8.72,  2.59,  3.70],
	[  1.38,  1.96,  0.90,  0.54,  2.34,  0.76,  1.45,  2.98,  0.49,  1.84,  1.68,  0.95,  0.44,  0.64,  0.65,  0.83,  1.27,  3.04,  2.20,  1.95],
	[  1.08,  4.71,  1.71,  1.22,  2.10,  1.45,  1.01,  2.95,  0.94,  1.66,  1.57,  1.66,  0.86,  0.77,  1.77,  1.66,  2.54,  3.76,  3.17,  2.34],
	[  5.01,  4.65,  1.22,  1.30,  7.09,  2.98,  2.95,  7.28,  1.80,  8.03,  4.71,  1.32,  0.90,  1.19,  1.79,  2.00,  3.42, 14.44,  6.09,  5.80],
	[  0.48,  0.66,  1.21,  1.25,  0.88,  0.49,  0.94,  1.80,  0.34,  0.71,  0.54,  0.33,  0.11,  0.64,  0.44,  0.67,  1.47,  1.70,  1.61,  1.38],
	[  2.64,  2.45,  0.92,  0.83,  3.10,  1.84,  1.66,  8.03,  0.71,  2.15,  2.72,  0.48,  0.76,  0.86,  1.18,  1.05,  2.05,  8.70,  2.30,  3.11],
	[  1.94,  2.68,  0.60,  0.74,  2.93,  1.68,  1.57,  4.71,  0.54,  2.72,  1.94,  0.68,  0.48,  0.66,  0.90,  1.17,  1.53,  5.78,  3.17,  2.95],
	[  0.83,  0.88,  1.22,  0.49,  1.28,  0.95,  1.66,  1.32,  0.33,  0.48,  0.68,  1.28,  0.23,  0.94,  0.35,  0.92,  2.05,  1.75,  1.51,  1.35],
	[  0.50,  0.71,  0.33,  0.20,  0.43,  0.44,  0.86,  0.90,  0.11,  0.76,  0.48,  0.23,  0.01,  0.36,  0.27,  0.28,  0.58,  1.39,  0.61,  0.80],
	[  0.75,  1.00,  0.46,  0.59,  1.24,  0.64,  0.77,  1.19,  0.64,  0.86,  0.66,  0.94,  0.36,  0.20,  0.60,  0.91,  1.45,  1.16,  1.83,  1.22],
	[  1.17,  0.53,  1.83,  1.93,  1.02,  0.65,  1.77,  1.79,  0.44,  1.18,  0.90,  0.35,  0.27,  0.60,  0.12,  0.73,  1.21,  1.81,  1.06,  1.91],
	[  1.11,  1.54,  1.26,  0.73,  1.32,  0.83,  1.66,  2.00,  0.67,  1.05,  1.17,  0.92,  0.28,  0.91,  0.73,  0.50,  1.81,  1.74,  1.38,  0.82],
	[  2.10,  1.63,  1.36,  0.98,  2.16,  1.27,  2.54,  3.42,  1.47,  2.05,  1.53,  2.05,  0.58,  1.45,  1.21,  1.81,  1.49,  3.78,  1.42,  2.12],
	[  5.35,  5.30,  1.62,  1.55,  8.72,  3.04,  3.76, 14.44,  1.70,  8.70,  5.78,  1.75,  1.39,  1.16,  1.81,  1.74,  3.78,  9.31,  6.57,  6.84],
	[  2.32,  6.10,  0.94,  0.89,  2.59,  2.20,  3.17,  6.09,  1.61,  2.30,  3.17,  1.51,  0.61,  1.83,  1.06,  1.38,  1.42,  6.57,  2.23,  1.99],
	[  2.44,  4.59,  0.82,  1.44,  3.70,  1.95,  2.34,  5.80,  1.38,  3.11,  2.95,  1.35,  0.80,  1.22,  1.91,  0.82,  2.12,  6.84,  1.99,  1.88],])
	antiparallelmatrix=array([[  0.79,  2.04,  0.77,  0.78,  2.98,  1.33,  1.36,  2.93,  0.80,  2.10,  1.49,  0.83,  0.42,  0.85,  1.52,  1.09,  1.91,  4.15,  2.75,  3.00],
	[  2.04,  6.45,  0.41,  0.77,  3.52,  1.40,  3.00,  3.39,  1.39,  1.87,  2.09,  0.63,  0.72,  0.96,  2.68,  1.84,  1.84,  3.97,  7.02,  4.22],
	[  0.77,  0.41,  0.27,  0.79,  1.17,  0.94,  1.86,  1.24,  1.60,  0.69,  0.82,  0.90,  0.33,  1.23,  1.64,  0.88,  1.78,  1.31,  1.33,  1.49],
	[  0.78,  0.77,  0.79,  0.55,  1.39,  0.77,  1.30,  1.96,  3.02,  1.11,  1.04,  1.31,  0.54,  1.28,  3.22,  1.18,  2.31,  2.57,  1.86,  2.44],
	[  2.98,  3.52,  1.17,  1.39,  2.60,  2.35,  2.58,  4.92,  1.76,  3.72,  3.66,  1.26,  1.37,  1.83,  2.38,  1.84,  2.45,  6.20,  5.84,  5.75],
	[  1.33,  1.40,  0.94,  0.77,  2.35,  0.61,  1.59,  2.05,  0.64,  1.41,  1.62,  0.99,  0.54,  1.02,  1.17,  0.98,  1.84,  3.05,  2.64,  2.72],
	[  1.36,  3.00,  1.86,  1.30,  2.58,  1.59,  1.46,  1.97,  1.28,  1.84,  1.98,  1.54,  0.80,  1.22,  1.80,  1.68,  3.20,  2.84,  2.46,  3.44],
	[  2.93,  3.39,  1.24,  1.96,  4.92,  2.05,  1.97,  3.20,  2.21,  3.98,  3.16,  1.08,  0.89,  1.70,  2.21,  1.53,  2.94,  7.63,  4.27,  5.36],
	[  0.80,  1.39,  1.60,  3.02,  1.76,  0.64,  1.28,  2.21,  0.79,  1.08,  1.02,  1.07,  0.32,  1.59,  1.01,  1.51,  2.62,  2.43,  2.12,  3.35],
	[  2.10,  1.87,  0.69,  1.11,  3.72,  1.41,  1.84,  3.98,  1.08,  1.53,  2.48,  0.80,  0.80,  1.27,  1.61,  1.28,  1.71,  4.47,  3.95,  3.56],
	[  1.49,  2.09,  0.82,  1.04,  3.66,  1.62,  1.98,  3.16,  1.02,  2.48,  1.47,  0.65,  0.86,  1.59,  1.40,  1.10,  1.83,  3.50,  2.56,  3.85],
	[  0.83,  0.63,  0.90,  1.31,  1.26,  0.99,  1.54,  1.08,  1.07,  0.80,  0.65,  0.63,  0.54,  1.57,  0.92,  1.28,  1.83,  1.57,  1.59,  2.10],
	[  0.42,  0.72,  0.33,  0.54,  1.37,  0.54,  0.80,  0.89,  0.32,  0.80,  0.86,  0.54,  0.19,  0.34,  0.66,  0.54,  0.95,  1.11,  1.30,  1.66],
	[  0.85,  0.96,  1.23,  1.28,  1.83,  1.02,  1.22,  1.70,  1.59,  1.27,  1.59,  1.57,  0.34,  0.76,  1.77,  1.38,  2.67,  2.24,  2.24,  2.55],
	[  1.52,  2.68,  1.64,  3.22,  2.38,  1.17,  1.80,  2.21,  1.01,  1.61,  1.40,  0.92,  0.66,  1.77,  0.95,  1.30,  2.59,  3.07,  3.95,  2.92],
	[  1.09,  1.84,  0.88,  1.18,  1.84,  0.98,  1.68,  1.53,  1.51,  1.28,  1.10,  1.28,  0.54,  1.38,  1.30,  0.84,  2.42,  2.37,  1.94,  2.83],
	[  1.91,  1.84,  1.78,  2.31,  2.45,  1.84,  3.20,  2.94,  2.62,  1.71,  1.83,  1.83,  0.95,  2.67,  2.59,  2.42,  2.39,  3.83,  3.51,  3.45],
	[  4.15,  3.97,  1.31,  2.57,  6.20,  3.05,  2.84,  7.63,  2.43,  4.47,  3.50,  1.57,  1.11,  2.24,  3.07,  2.37,  3.83,  4.91,  5.95,  7.39],
	[  2.75,  7.02,  1.33,  1.86,  5.84,  2.64,  2.46,  4.27,  2.12,  3.95,  2.56,  1.59,  1.30,  2.24,  3.95,  1.94,  3.51,  5.95,  5.17,  7.60],
	[  3.00,  4.22,  1.49,  2.44,  5.75,  2.72,  3.44,  5.36,  3.35,  3.56,  3.85,  2.10,  1.66,  2.55,  2.92,  2.83,  3.45,  7.39,  7.60,  4.00],])

	# if num>0:
	# 	return None
	#printInfo('Dealing with {:04} with {:.0f} seqs, resnum {}({})'.format(num,seqnum,len(seq),name))

	ignoreraise=0
	ratios=['all',0.1,0.2,0.3,0.5,0.75,1,2,3,4,5,10]
	paraprefer=zeros((len(seq),len(seq),1))
	antiparaprefer=zeros((len(seq),len(seq),1))
	posdif=zeros((len(seq),len(seq),1))
	for i in range(len(seq)):
		for j in range(len(seq)):
			paraprefer[i,j]=parallelmatrix[resmap[seq[i]],resmap[seq[j]]].mean()
			antiparaprefer[i,j]=antiparallelmatrix[resmap[seq[i]],resmap[seq[j]]].mean()
			posdif[i,j]=abs(i-j)
	secondary=zeros((len(seq),len(seq),6))
	for i in range(len(seq)):
		secondary[i,:,:3]=deepcnf[i]
		secondary[:,i,3:]=deepcnf[i]
	# for i in range(len(matrixs)):
	for i in range(1):
		if matrixs[i] is None:
			continue
		if os.path.exists('feature/{}-{}-{}.npy'.format(proname,ratios[i],method)):
			continue
		#printInfo('Dealing with {} with {}'.format(proname,ratios[i]))
		if ignoreraise:
			try:
				result=calculate(matrixs[i],
					method,
					output=True, plotfig=True, tag=ratios[i])
			except:
				print proname,'Wrong\n\n\n\n'
		else:
			result=calculate(matrixs[i],
				method,
				output=True, plotfig=True, tag=ratios[i])
		feature=concatenate((result,paraprefer,antiparaprefer,posdif,secondary,ones((len(seq),len(seq),1))*seqnum/resnum,ones((len(seq),len(seq),1))*matrixs[i].std()),axis=2)
		save('feature/{}-{}-{}.npy'.format(proname,ratios[i],method),feature)


def runresult(use_parallel=False, *args, **kwargs):
	if use_parallel:
		from multiprocessing.pool import Pool
		pool = Pool(8)
		result=[i for i in pool.map(dealwith,packzip(*args)) if i is not None]
		pool.close()
		pool.join()
	else:
		result=[i for i in map(dealwith,packzip(*args)) if i is not None]
	return result
