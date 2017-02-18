import numpy as npy
import mxnet as mx
import logging

from symbols.symbol_dh import DHMidLayer,DHLossLayer
from common.data import SiftSmallIter

batchsize=50

opProj1=DHMidLayer(96,0.0001,0.0001)
opProj2=DHMidLayer(64,0.0001,0.0001)
opOut=DHLossLayer(0.001)

data = mx.symbol.Variable('data')
lm1=opProj1(data=data, name='lm1')
lm2=opProj2(data=lm1, name="lm2")
netDH=opOut(data=lm2)
ex = netDH.simple_bind(ctx=mx.cpu(), data=(batchsize, 128))

listArgs = dict(zip(netDH.list_arguments(), ex.arg_arrays))

for arg in listArgs:
    data = listArgs[arg]
    if 'weight' in arg:
        data[:] = mx.random.uniform(-0.1, 0.1, data.shape)
    if 'bias' in arg:
        data[:] = 0

dataPath="E:\\DevProj\\Datasets\\SIFT1M\\siftsmall"
trainIter, valIter=SiftSmallIter(dataPath,21000,4000,batchsize)

learning_rate=0.01
for ii in range(200):
    print "Deep Hash Training at iteration "+str(ii)
    trainbatch=trainIter.next()
    listArgs['data'][:] = trainbatch.data[0]
    ex.forward(is_train=True)
    ex.backward()
    for arg, grad in zip(ex.arg_arrays, ex.grad_arrays):
        arg[:] -= learning_rate * (grad / batchsize)
        
xx=ex.outputs[0].asnumpy()

