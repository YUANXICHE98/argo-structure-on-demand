import argparse, json, sys
from pathlib import Path
import numpy as np, torch
import torch.nn.functional as F
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from src.data.camels_loader import CAMELSSpatialDataset, load_camels_us
from src.models.projection_space_routing_hypergraph import ProjectionSpaceRoutingHyperNet


def parse_args():
    p=argparse.ArgumentParser(); p.add_argument('--checkpoint',type=Path,required=True); p.add_argument('--device',default='cpu'); p.add_argument('--max-batches',type=int,default=None); p.add_argument('--top-frac',type=float,default=0.1); p.add_argument('--output-json',type=Path,default=ROOT/'results/structure_diagnosis.json'); p.add_argument('--output-md',type=Path,default=ROOT/'results/structure_diagnosis.md'); return p.parse_args()

def make_model(ckpt,data,device):
    m=ProjectionSpaceRoutingHyperNet(128,data['F'],data['D_s']).to(device); m.load_state_dict(torch.load(ckpt,map_location=device,weights_only=True)); m.eval(); return m

def summary(x):
    a=np.asarray(x,dtype=float); return {'mean':float(a.mean()) if a.size else 0,'std':float(a.std()) if a.size else 0,'min':float(a.min()) if a.size else 0,'max':float(a.max()) if a.size else 0}

def jaccard(a,b,frac):
    x=a.flatten(); y=b.flatten(); k=max(1,int(x.numel()*frac)); ix=torch.topk(x,k).indices; iy=torch.topk(y,k).indices; mx=torch.zeros(x.numel(),device=x.device,dtype=torch.bool); my=torch.zeros_like(mx); mx[ix]=1; my[iy]=1; return float(1-((mx&my).sum().float()/((mx|my).sum().float().clamp(min=1))).cpu())

def collect(model, forcing, static, top_frac):
    b,n,t,f=forcing.shape; hs=model.static_encoder(static); ht=model.temporal_encoder(forcing.reshape(b*n,t,f), hs.reshape(b*n,-1)).reshape(b,n,-1); out=[]
    for i in range(b):
        sv,views=model._build_views(ht[i],hs[i]); spaces=torch.stack([g(v)[0] for g,v in zip(model.projection_generators,views)]); sm=sv.mean(0); tm=views[0].mean(0); dis=torch.stack([torch.abs(v.mean(0)-sm) for v in views]).mean(0); probs,beta=model.router(sm,tm,dis); flat=F.normalize(spaces.reshape(spaces.shape[0],-1),dim=-1)
        for a in range(spaces.shape[0]):
            for c in range(a+1,spaces.shape[0]): out.append({'cosine':float(1-torch.dot(flat[a],flat[c]).cpu()),'jaccard':jaccard(spaces[a],spaces[c],top_frac),'routing':[float(v) for v in probs.cpu()],'beta':float(beta.cpu())})
    return out

def main():
    args=parse_args(); device=args.device; data=load_camels_us(ROOT/'data/camels_us'); loader=torch.utils.data.DataLoader(CAMELSSpatialDataset(data,26,'test',max_basins=data['N']),batch_size=8); model=make_model(args.checkpoint,data,device); rows=[]
    with torch.no_grad():
        for i,batch in enumerate(loader):
            if args.max_batches is not None and i>=args.max_batches: break
            forcing,static,_,_=[x.to(device) for x in batch]; rows += collect(model,forcing,static,args.top_frac)
    routing=np.asarray([r['routing'] for r in rows]) if rows else np.zeros((0,3)); payload={'checkpoint':str(args.checkpoint),'sample_pairs':len(rows),'pairwise_cosine_distance':summary([r['cosine'] for r in rows]),'pairwise_jaccard_distance':summary([r['jaccard'] for r in rows]),'routing_mean':routing.mean(0).tolist() if len(routing) else [],'routing_std':routing.std(0).tolist() if len(routing) else [],'beta':summary([r['beta'] for r in rows])}
    args.output_json.parent.mkdir(parents=True,exist_ok=True); args.output_json.write_text(json.dumps(payload,indent=2)); args.output_md.write_text('# Structure diagnosis\n\n```json\n'+json.dumps(payload,indent=2)+'\n```\n'); print(json.dumps(payload,indent=2))
if __name__=='__main__': main()
