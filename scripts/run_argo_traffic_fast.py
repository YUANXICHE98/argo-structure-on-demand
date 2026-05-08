import json, sys, time
from pathlib import Path
import numpy as np, torch
import torch.nn as nn
import torch.nn.functional as F
ROOT=Path(__file__).resolve().parents[1]; RESULT_DIR=ROOT/'results'/'argo_traffic_fast'; RESULT_DIR.mkdir(parents=True,exist_ok=True)

def load_data():
    data_path=ROOT/'data/traffic_datasets/real/metr_la_data.npy'; adj_path=ROOT/'data/traffic_datasets/real/metr_la_adj.npy'
    if not data_path.exists() or not adj_path.exists(): raise FileNotFoundError('Place METR-LA arrays at data/traffic_datasets/real/metr_la_data.npy and metr_la_adj.npy')
    data=np.load(data_path).T; adj=np.load(adj_path); T,N=data.shape; tr=int(T*.7); va=int(T*.85); mean,std=data[:tr].mean(),data[:tr].std()+1e-8; data=(data-mean)/std
    X=[];Y=[]
    for i in range(len(data)-24+1): X.append(data[i:i+12]); Y.append(data[i+12:i+24])
    X=np.array(X); Y=np.array(Y)
    return {'train':(X[:tr],Y[:tr]),'val':(X[tr:va],Y[tr:va]),'test':(X[va:],Y[va:]),'adj':adj,'N':N}

class FastARGO(nn.Module):
    def __init__(self,N,D=64,E=32,M=3,use_orthogonal_init=True):
        super().__init__(); self.N=N; self.M=M; self.E=E; self.proj=nn.Linear(1,D); self.gru=nn.GRU(D,D,batch_first=True); self.space_projs=nn.ModuleList([nn.Linear(D,D) for _ in range(M)]); self.router=nn.Linear(D*2,M); self.out=nn.Linear(D,1); self.prior_logit=nn.Parameter(torch.tensor(1.0)); self.register_buffer('prior',None)
        self.queries=nn.ParameterList()
        for _ in range(M):
            q=torch.randn(E,D)*.1
            if use_orthogonal_init:
                pad=torch.zeros(D,D); pad[:E]=q; q=torch.linalg.qr(pad).Q[:E]
            self.queries.append(nn.Parameter(q))
    def set_prior(self,adj):
        a=(torch.as_tensor(adj)>0.1).float()+torch.eye(self.N); self.prior=a/a.sum(1,keepdim=True).clamp(min=1e-8)
    def forward(self,x):
        B,N,T,_=x.shape; h=self.proj(x.permute(0,2,1,3)).reshape(B*N,T,-1); _,h=self.gru(h); h=h.reshape(B,N,-1); probs=F.softmax(self.router(torch.cat([h.mean(1),h.std(1)],-1)),-1); z_prior=torch.matmul(self.prior.to(h.device).unsqueeze(0),h) if self.prior is not None else torch.zeros_like(h); Hs=[]
        for s in range(self.M):
            hs=F.normalize(self.space_projs[s](h),dim=-1); q=F.normalize(self.queries[s],dim=-1); scores=torch.matmul(hs,q.T); k=min(N,max(4,int(N*.05))); vals,idx=torch.topk(scores,k,dim=1); w=F.softmax(vals/.7,dim=1); H=torch.zeros_like(scores); H.scatter_(1,idx,w); Hs.append(H)
        Hmix=(torch.stack(Hs,1)*probs[:,:,None,None]).sum(1); edge=torch.bmm(Hmix.transpose(1,2),h); z_learn=torch.bmm(Hmix,edge); alpha=torch.sigmoid(self.prior_logit); return self.out(alpha*z_prior+(1-alpha)*z_learn).squeeze(-1), probs

class BaselineGRU(nn.Module):
    def __init__(self,N,D=64): super().__init__(); self.proj=nn.Linear(1,D); self.gru=nn.GRU(D,D,batch_first=True); self.out=nn.Linear(D,1)
    def forward(self,x):
        B,N,T,_=x.shape; h=self.proj(x.permute(0,2,1,3)).reshape(B*N,T,-1); _,h=self.gru(h); return self.out(h.reshape(B,N,-1)).squeeze(-1), None

def train(name,model,data,device,epochs=30):
    model.to(device); 
    if hasattr(model,'set_prior'): model.set_prior(data['adj'])
    opt=torch.optim.Adam(model.parameters(),lr=1e-3); X=torch.tensor(data['train'][0],dtype=torch.float32); Y=torch.tensor(data['train'][1],dtype=torch.float32); vX=torch.tensor(data['val'][0],dtype=torch.float32); vY=torch.tensor(data['val'][1],dtype=torch.float32); best=1e9; state=None
    for ep in range(epochs):
        model.train(); perm=torch.randperm(len(X))
        for i in range(0,len(X),64):
            idx=perm[i:i+64]; x=X[idx].permute(0,2,1).unsqueeze(-1).to(device); y=Y[idx].mean(1).to(device); opt.zero_grad(); pred,_=model(x); F.l1_loss(pred,y).backward(); opt.step()
        model.eval(); val=0
        with torch.no_grad():
            for i in range(0,len(vX),64):
                x=vX[i:i+64].permute(0,2,1).unsqueeze(-1).to(device); y=vY[i:i+64].mean(1).to(device); pred,_=model(x); val+=F.l1_loss(pred,y).item()*len(x)
        val/=len(vX); print(name,ep,val,flush=True)
        if val<best: best=val; state={k:v.cpu().clone() for k,v in model.state_dict().items()}
    model.load_state_dict(state); return model

def evaluate(model,data,device):
    model.eval(); X=torch.tensor(data['test'][0],dtype=torch.float32); Y=torch.tensor(data['test'][1],dtype=torch.float32); preds=[]; tgts=[]; routes=[]
    with torch.no_grad():
        for i in range(0,len(X),64):
            x=X[i:i+64].permute(0,2,1).unsqueeze(-1).to(device); y=Y[i:i+64].mean(1).to(device); pred,p=model(x); preds.append(pred.cpu()); tgts.append(y.cpu());
            if p is not None: routes.append(p.cpu())
    p=torch.cat(preds).numpy().ravel(); y=torch.cat(tgts).numpy().ravel(); out={'MAE':round(float(np.mean(np.abs(p-y))),4),'RMSE':round(float(np.sqrt(np.mean((p-y)**2))),4)}
    if routes: out['routing_mean']=torch.cat(routes).mean(0).tolist(); out['prior_alpha']=float(torch.sigmoid(model.prior_logit.detach()).cpu())
    return out

def main():
    device='mps' if torch.backends.mps.is_available() else 'cpu'; data=load_data(); results={}
    for name,model in [('Full_Ortho',FastARGO(data['N'],use_orthogonal_init=True)),('NoOrtho',FastARGO(data['N'],use_orthogonal_init=False)),('SingleSpace',FastARGO(data['N'],M=1,use_orthogonal_init=False)),('Baseline',BaselineGRU(data['N']))]:
        t=time.time(); model=train(name,model,data,device); m=evaluate(model,data,device); m['time_s']=round(time.time()-t,1); results[name]=m; print(name,m,flush=True)
    (RESULT_DIR/'results.json').write_text(json.dumps(results,indent=2)); print(json.dumps(results,indent=2))
if __name__=='__main__': main()
