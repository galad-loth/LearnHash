import mxnet as mx
import numpy as npy

class DHMidLayer(mx.operator.NumpyOp):
    def __init__(self, proj_dim, lambda2, lambda3):
        super(DHMidLayer, self).__init__(True)
        self.lambda2=lambda2
        self.lambda3=lambda3
        self.proj_dim=proj_dim
        
    def list_arguments(self):
        return ['data','weights','bias']
        
    def list_outputs(self):
        return ['output']
        
    def infer_shape(self, in_shape):
        data_shape=in_shape[0]
        weight_shape=(in_shape[0][1],self.proj_dim)
        bias_shape=(1,self.proj_dim)
        output_shape=(in_shape[0][0],self.proj_dim)
        return [data_shape,weight_shape,bias_shape],[output_shape]
        
    def forward(self, in_data, out_data):
        x = in_data[0]
        w =in_data[1]
        b =in_data[2]
        y = out_data[0]
        y1=npy.dot(x,w)+b
        y[:]=(npy.exp(-y1*2)-1)/(npy.exp(-y1*2)+1)
        
    def backward(self, out_grad, in_data, out_data, in_grad):
        dx=in_grad[0]
        dw=in_grad[1]
        db=in_grad[2]
        
        x = in_data[0]
        w =in_data[1]
        b =in_data[2]
        y = out_data[0]
        
        y2=y*y
        dxout=out_grad[0]
        dxout1=dxout*(1-y2)
        dx[:]=npy.dot(dxout1,w.T)
        dw0=npy.dot(x.T,dxout1) 
        wTw=npy.dot(w.T,w)
        dw[:]=dw0+self.lambda3*w+self.lambda2*npy.dot(w,wTw-npy.eye(w.shape[1]))
        db[:]=npy.sum(dxout1, axis=0)+self.lambda3*b        
     
        
class DHLossLayer(mx.operator.NumpyOp):
    def __init__(self, lambda1):
        super(DHLossLayer, self).__init__(False)
        self.lambda1=lambda1
        
    def list_arguments(self):
        return ['data']
        
    def list_outputs(self):
        return ['outputs']
        
    def infer_shape(self, in_shape):
        data_shape=in_shape[0]
        return [data_shape],[data_shape]
        
    def forward(self, in_data, out_data):
        x=in_data[0]
        y=out_data[0]
        y[x>=0]=1
        y[x<0]=-1     
        
        
    def backward(self, out_grad, in_data, out_data, in_grad):
        x = in_data[0]
        y = out_data[0]        
        dx=in_grad[0]
        
        dx[:]=(1-self.lambda1)*x-y
        
if __name__=="__main__":
    opProj1=DHMidLayer(96,0.0001,0.0001)
    opProj2=DHMidLayer(64,0.0001,0.0001)
    opOut=DHLossLayer(0.001)
    
    data = mx.symbol.Variable('data')
    lm1=opProj1(data=data, name='lm1')
    lm2=opProj2(data=lm1, name="lm2")
    netDH=opOut(data=lm2)
    ex = netDH.simple_bind(ctx=mx.cpu(), data=(50, 128))