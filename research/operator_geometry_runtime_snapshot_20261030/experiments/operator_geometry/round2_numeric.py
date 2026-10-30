"""Vectorized average-tie Spearman correlations for independent row choices."""
import numpy as np


def row_ranks(values):
    values=np.asarray(values);order=np.argsort(values,axis=1,kind='stable')
    sorted_=np.take_along_axis(values,order,axis=1);n,k=values.shape;col=np.arange(k)[None,:]
    starts=np.concatenate([np.ones((n,1),bool),sorted_[:,1:]!=sorted_[:,:-1]],axis=1)
    ends=np.concatenate([sorted_[:,:-1]!=sorted_[:,1:],np.ones((n,1),bool)],axis=1)
    left=np.maximum.accumulate(np.where(starts,col,0),axis=1)
    right=np.minimum.accumulate(np.where(ends,col,k)[:,::-1],axis=1)[:,::-1]
    result=np.empty(values.shape,float);np.put_along_axis(result,order,(left+right)/2,axis=1)
    return result


def row_spearman(a,b):
    a=row_ranks(a);b=row_ranks(b);a-=a.mean(1,keepdims=True);b-=b.mean(1,keepdims=True)
    den=np.sqrt(np.square(a).sum(1)*np.square(b).sum(1))
    result=np.full(len(a),np.nan);valid=den>0
    result[valid]=(a[valid]*b[valid]).sum(1)/den[valid]
    return result
