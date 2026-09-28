from graph_bench import *
torch.manual_seed(11);G=graph(np.argwhere(Y>0),11);a=init_model(G,'TxGNN');b=copy.deepcopy(a);enable_cuda_encoder(b)
a.eval();b.eval();sa=predict(a,G,'TxGNN');sb=gpu_decoder(b,G,b(G,None,return_h=True)).cpu();err=float((sa-sb).abs().max());assert err<2e-5,err
sa.sum().backward();sb.sum().backward();errors={}
for (n,p),(nn,q) in zip(a.named_parameters(),b.named_parameters()):
 if p.grad is not None:
  absolute=float((p.grad-q.grad.cpu()).abs().max());scale=float(p.grad.abs().max());errors[n]=absolute/max(scale,1e-8)
assert max(errors.values())<1e-4,max(errors.values())
(W/'protocol/gpu_encoder_equivalence.json').write_text(json.dumps({'max_score_error':err,'max_relative_gradient_error':max(errors.values()),'status':'passed','method':'Exact mean aggregation expressed by relation-wise normalized sparse adjacency; original decoder unchanged'},indent=2),encoding='utf8');print(err,max(errors.values()),flush=True)
