# -*- coding: utf-8 -*-
import sys
from numpy import *
import os
from screen_tool import *

def checkandmake(x):
	import os
	if not os.path.exists(x):
		os.mkdir(x)

proname=sys.argv[1]

def getinfo():
	f=open('../sequence/{}'.format(proname),'r')
	while 1:
		temp=f.readline()
		if temp=='':
			break
		fastas=f.readline().strip()
	f.close()

	resnum=len(fastas)
	f=open('../Spider3/{}.spd33'.format(proname),'r')
	lines=[l.strip() for l in f if not l.startswith('#') and l.strip()!='']
	f.close()
	deepcnf=array([[float(j) for j in ii.split()[-3:]] for ii in lines])
	f=open('../ccmpred/{}.aln'.format(proname))
	seqnum=f.read().strip().count('\n')+1
	f.close()
	matrixs=[loadtxt('../deepconpred2/{}_contactmap.txt'.format(proname))]
	matrixs=[i if i is None else (i+i.T)/2. for i in matrixs]
	return [proname,int(resnum),array(list(fastas)),deepcnf,seqnum,matrixs,'NL']


use_parallel=0
#checkandmake('feature')
method='NL'
info=getinfo()
result=dealwith(info)

##make reige features out of picture runresult in (screen_tool.py)
