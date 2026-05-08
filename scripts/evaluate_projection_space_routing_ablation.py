import argparse, json, sys
from pathlib import Path
import numpy as np, torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data.camels_loader import CAMELSSpatialDataset, load_camels_us
from src.models.projection_space_routing_hypergraph import ProjectionSpaceRoutingHyperNet
from scripts.run_weekly_projection_space_routing import evaluate


def parse_args():
    p=argparse.ArgumentParser(); p.add_argument('--checkpoint', type=Path, required=True); p.add_argument('--device', default='cpu'); p.add_argument('--output-json', type=Path, default=ROOT/'results/router_interventions.json'); p.add_argument('--output-md', type=Path, default=ROOT/'results/router_interventions.md'); return p.parse_args()


def make_model(ckpt, data, device):
    m=ProjectionSpaceRoutingHyperNet(128, data['F'], data['D_s']).to(device); m.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True)); m.eval(); return m


def select_structure(model, shared, spaces, static_mean, temporal_mean, disagreement, strategy):
    learned, beta = model.router(static_mean, temporal_mean, disagreement)
    if strategy == 'learned': probs = learned
    elif strategy == 'uniform_average': probs = torch.full_like(learned, 1/model.num_projection_spaces)
    elif strategy == 'shuffled_space': probs = learned[torch.randperm(len(learned), device=learned.device)]
    elif strategy.startswith('fixed_space'):
        idx=int(strategy.replace('fixed_space','')); probs=torch.zeros_like(learned); probs[idx]=1
    elif strategy == 'static_only': probs=torch.zeros_like(learned); beta=torch.zeros_like(beta)
    else: raise ValueError(strategy)
    return model._mix_projection_spaces(shared, spaces, probs, beta)


def forward_strategy(model, forcing, static, strategy):
    b,n,t,f=forcing.shape; hs=model.static_encoder(static); ht=model.temporal_encoder(forcing.reshape(b*n,t,f), hs.reshape(b*n,-1)).reshape(b,n,-1); outs=[]
    for i in range(b):
        sv, views=model._build_views(ht[i], hs[i]); shared,_=model.shared_generator(sv); spaces=torch.stack([g(v)[0] for g,v in zip(model.projection_generators, views)])
        sm=sv.mean(0); tm=views[0].mean(0); dis=torch.stack([torch.abs(v.mean(0)-sm) for v in views]).mean(0)
        inc=select_structure(model, shared, spaces, sm, tm, dis, strategy); z=ht[i]
        for conv in model.hg_convs: z=conv(z, inc)
        outs.append(model.predictor(torch.cat([z,hs[i]],-1)).squeeze(-1))
    return torch.stack(outs)


def eval_strategy(model, loader, device, strategy):
    model.eval(); preds=[]; obs=[]; masks=[]
    with torch.no_grad():
        for forcing, static, target, mask in loader:
            preds.append(forward_strategy(model, forcing.to(device), static.to(device), strategy).cpu()); obs.append(target); masks.append(mask)
    p=torch.cat(preds).numpy().ravel(); y=torch.cat(obs).numpy().ravel(); m=torch.cat(masks).numpy().ravel()>0; p,y=p[m],y[m]
    return {'nse':float(1-np.sum((p-y)**2)/(np.sum((y-y.mean())**2)+1e-10)), 'rmse':float(np.sqrt(np.mean((p-y)**2))), 'corr':float(np.corrcoef(p,y)[0,1]), 'n':int(len(p))}


def main():
    args=parse_args(); device=args.device; torch.manual_seed(42); np.random.seed(42); data=load_camels_us(ROOT/'data/camels_us'); loader=torch.utils.data.DataLoader(CAMELSSpatialDataset(data,26,'test',max_basins=data['N']), batch_size=8); model=make_model(args.checkpoint,data,device)
    strategies=['learned','uniform_average','shuffled_space','fixed_space0','fixed_space1','fixed_space2','static_only']; results={s:eval_strategy(model,loader,device,s) for s in strategies}
    args.output_json.parent.mkdir(parents=True, exist_ok=True); args.output_json.write_text(json.dumps({'checkpoint':str(args.checkpoint),'results':results},indent=2));
    base=results['learned']['nse']; lines=['# Router interventions','','| strategy | NSE | delta | RMSE |','|---|---:|---:|---:|']+[f"| {s} | {r['nse']:.4f} | {r['nse']-base:+.4f} | {r['rmse']:.4f} |" for s,r in results.items()]
    args.output_md.write_text('\n'.join(lines)); print(json.dumps(results,indent=2))
if __name__=='__main__': main()
