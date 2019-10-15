# -*- coding: utf-8 -*-

from __future__ import print_function

import os

import tensorflow as tf

from numpy import *

import sys

from sklearn import metrics

import matplotlib
matplotlib.use('Agg')
from matplotlib.pyplot import *
import matplotlib.pyplot as plt
plt.rc('axes', prop_cycle=(cycler('color', ['#16a085', '#2980b9','#c0392b','#7f8c8d', '#8e44ad','#2ecc71','#2c3e50','#d35400','#bdc3c7',"#f39c12"])))

proname=sys.argv[1]

def getdata(file_path):
	data=[]
	for serialized_example in tf.python_io.tf_record_iterator(file_path):
		example = tf.train.Example()
		example.ParseFromString(serialized_example)

		length = example.features.feature['length'].int64_list.value[0]
		proname = example.features.feature['proname'].bytes_list.value[0]

		seq=fromstring(example.features.feature['seq'].bytes_list.value[0])
		seql=zeros((length,20))
		seql[seq!=20,seq[seq!=20].astype(int)]=1.
		seql[seq==20,:]=1./20

		ccmpred = fromstring(example.features.feature['ccmpred'].bytes_list.value[0])
		ccmpred.resize((length,length,1))
		deepconpred2 = fromstring(example.features.feature['deepconpred2'].bytes_list.value[0])
		deepconpred2.resize((length,length,1))
		ridge2 = fromstring(example.features.feature['ridge2'].bytes_list.value[0])
		ridge2.resize((length,length,4))
		ridge2=ridge2[:,:,[0,3]]
		posdif2 = fromstring(example.features.feature['posdif2'].bytes_list.value[0])
		posdif2.resize((length,length,1))
		casefeature2 = fromstring(example.features.feature['casefeature2'].bytes_list.value[0])
		casefeature2.resize((length,length,2))
		spd = fromstring(example.features.feature['spd'].bytes_list.value[0])
		spd.resize((length,3))
		seqnum = example.features.feature['seqnum'].int64_list.value[0]
		if not length==ccmpred.shape[0]==ccmpred.shape[1]==spd.shape[0]==spd.shape[0]==ridge2.shape[0]==ridge2.shape[1]==deepconpred2.shape[0]==deepconpred2.shape[1]:
			raise ValueError('Wrong shape')
		feature0d=array([length,seqnum]).astype(float32)
		feature1d=concatenate((seql,spd),axis=1).astype(float32)
		feature2d=concatenate((ccmpred,ridge2,deepconpred2),axis=2).astype(float32)
		data.append([feature0d,feature1d,feature2d])
	return data

fold='./model/'
parafolds=sorted([os.path.join(fold,i) for i in os.listdir(fold) if os.path.isdir(os.path.join(fold,i)) and i.startswith('Para')])
epochs=[]
for parafold in parafolds:
	a=sorted([int(i.split('.')[1].split('-')[1]) for i in os.listdir(parafold) if i.startswith("Para") and i.endswith('.meta')])
	epochs.append(a)
if not all([i==epochs[0] for i in epochs]):
	print(fold,'Not all the same')
for datafile in ['./tfrecords/{}.tfrecords'.format(proname)]:
	data=getdata(datafile)
	traintps=[loadtxt(i) for i in sorted([os.path.join(fold,i) for i in os.listdir(fold) if i.startswith('tps')])]
	traintps=[i.reshape(i.shape[0],3,-1) for i in traintps]
	trainfs=[i[:,1]*2/(i[:,0]+i[:,2]) for i in traintps]
	cutoffs=[i.argmax(1) for i in trainfs]
	trainfs=[i.max(1) for i in trainfs]
	bestepochids=[argmax(i) for i in trainfs]
	traintp1=[traintps[i][bestepochids[i]] for i in range(5)]
	traintp1=array(traintp1)
	traintp1=traintp1.sum(axis=0)
	trainf1=traintp1[1]*2/(traintp1[0]+traintp1[2])
	trainf1_final=trainf1
	cutoff1_final=trainf1.argmax(axis=0)
	testpredsum=[]
	for crossfold in range(len(parafolds)):
		epoch=bestepochids[crossfold]
		path=parafolds[crossfold]
		tf.reset_default_graph()
		saver = tf.train.import_meta_graph('{}/Para.ckpt-{}.meta'.format(path,epoch))  #for循环中 导入最好的5个模型
		graph = tf.get_default_graph()
		config = tf.ConfigProto()
		config.gpu_options.allow_growth=True
		with tf.Session(config=config) as sess:
			saver.restore(sess, '{}/Para.ckpt-{}'.format(path,epoch))
			x=graph.get_operation_by_name('x').outputs[0]
			istrain=graph.get_operation_by_name('istrain').outputs[0]
			realpred=graph.get_operation_by_name('Final/Sigmoid').outputs[0]
			testpred=[]
			for case in range(len(data)):
				sys.stdout.flush()
				f0d,f1d,f2d=data[case]
				n=f1d.shape[0]
				predresult=sess.run(realpred,feed_dict={x: concatenate([
															f2d,
															tile(f1d[newaxis],[n,1,1]),
															tile(f1d[:,newaxis],[1,n,1]),
															tile(f0d[newaxis,newaxis],[n,n,1]),
															],axis=2)[newaxis], istrain:False,}).reshape((n,n))
				predresult=(predresult+predresult.T)/2.
				testpred.append(predresult)
			if testpredsum==[]:
				testpredsum=testpred
			else:
				for i in range(len(data)):
					testpredsum[i]+=testpred[i]

	testpredsum=[i/len(parafolds) for i in testpredsum]
	for i in range(len(data)):
		savetxt('./result/{}_result.txt'.format(proname),testpredsum[i])
        subplot(111)
        imshow(testpredsum[i],cmap='gray_r')
        savefig('./result/{}_map.png'.format(proname))
        close('all')
