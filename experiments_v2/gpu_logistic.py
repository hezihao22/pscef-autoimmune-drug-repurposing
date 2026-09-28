"""Full-dimensional balanced logistic regression, same liblinear L2 objective.
Four regularization strengths optimized together on local GPU. Intercept is
penalized like liblinear's default synthetic intercept feature.
"""
import numpy as np,torch
from sklearn.linear_model import LogisticRegression
from scipy.stats import spearmanr
torch.set_num_threads(4)
def gpu_fit_predict(xf,yy,xq,cs=(.1,1,10,100)):
 device='cuda';x=torch.as_tensor(np.asarray(xf,dtype='float32'),device=device);q=torch.as_tensor(np.asarray(xq,dtype='float32'),device=device);n,d=x.shape
 t=torch.as_tensor(yy.astype('float32'),device=device)[:,None];counts=np.bincount(yy);weight=torch.as_tensor(n/(2*counts[yy]),dtype=torch.float32,device=device)[:,None]
 c=torch.tensor(cs,dtype=torch.float32,device=device);w=torch.zeros((d,len(cs)),device=device,requires_grad=True);b=torch.zeros(len(cs),device=device,requires_grad=True)
 opt=torch.optim.LBFGS([w,b],lr=1,max_iter=350,max_eval=500,tolerance_grad=1e-7,tolerance_change=1e-9,history_size=30,line_search_fn='strong_wolfe')
 def closure():
  opt.zero_grad();z=x@w+b;loss=(torch.nn.functional.binary_cross_entropy_with_logits(z,t.expand_as(z),reduction='none')*weight).mean(0).sum()+((w.square().sum(0)+b.square())/(2*c*n)).sum();loss.backward();return loss
 opt.step(closure)
 with torch.no_grad():result=(q@w+b).cpu().numpy()
 return result
