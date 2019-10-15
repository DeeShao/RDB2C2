# -*- coding: utf-8 -*-

import os
import sys
from numpy import *
from numpy.linalg import *
import tensorflow as tf

proname=sys.argv[1]

l2l={"A":0,"R":1,"N":2,"D":3,"C":4,"E":5,"Q":6,"G":7,"H":8,"I":9,"L":10,"K":11,"M":12,"F":13,"P":14,"S":15,"T":16,"W":17,"Y":18,"V":19,"X":20}
def getinfo(proname):
	f=open('../sequence/{}'.format(proname),'r')
	seq=''.join(f.read().strip().split('\n')[1:])
	f.close()
	seq=array([l2l[i] for i in seq],dtype=float)
	ccmpred=loadtxt('../ccmpred/{}.ccmpred'.format(proname))
	n=len(seq)
	deepconpred2=loadtxt('../deepconpred2/{}_contactmap.txt'.format(proname))
	ridge=load('../ridge/feature/{}-all-NL.npy'.format(proname))
	ridge2=ridge[:,:,[0,1,2,3]]
	posdif2=ridge[:,:,[10]]
	casefeature2=ridge[:,:,[17,18]]
	f=open('../Spider3/{}.spd33'.format(proname),'r')
	lines=[l.strip() for l in f if not l.startswith('#') and l.strip()!='']
	f.close()
	spd=array([[float(j) for j in ii.split()[-3:]] for ii in lines])
	f=open('../ccmpred/{}.aln'.format(proname))
	seqnum=f.read().strip().count('\n')+1 #861
	f.close()
	return n,proname,seq,ccmpred,deepconpred2,ridge2,posdif2,casefeature2,spd,seqnum


writer = tf.python_io.TFRecordWriter("./tfrecords/{}.tfrecords".format(proname))
n,proname,seq,ccmpred,deepconpred2,ridge2,posdif2,casefeature2,spd,seqnum=getinfo(proname)
example = tf.train.Example(features=tf.train.Features(feature={
	'length': tf.train.Feature(int64_list=tf.train.Int64List(value=[n])),
	"proname": tf.train.Feature(bytes_list=tf.train.BytesList(value=[proname])),
	'seq': tf.train.Feature(bytes_list=tf.train.BytesList(value=[seq.astype(float64).tobytes()])),
	'ccmpred': tf.train.Feature(bytes_list=tf.train.BytesList(value=[ccmpred.astype(float64).tobytes()])),
	'deepconpred2': tf.train.Feature(bytes_list=tf.train.BytesList(value=[deepconpred2.astype(float64).tobytes()])),
	'ridge2': tf.train.Feature(bytes_list=tf.train.BytesList(value=[ridge2.astype(float64).tobytes()])),
	'posdif2': tf.train.Feature(bytes_list=tf.train.BytesList(value=[posdif2.astype(float64).tobytes()])),
	'casefeature2': tf.train.Feature(bytes_list=tf.train.BytesList(value=[casefeature2.astype(float64).tobytes()])),
	'spd': tf.train.Feature(bytes_list=tf.train.BytesList(value=[spd.astype(float64).tobytes()])),
	'seqnum': tf.train.Feature(int64_list=tf.train.Int64List(value=[seqnum])),
	}))
writer.write(example.SerializeToString())
writer.close()
