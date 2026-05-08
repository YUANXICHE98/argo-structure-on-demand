import argparse, json, os, sys, time
from pathlib import Path
import numpy as np, torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data.camels_loader import CAMELSSpatialDataset, load_camels_us
from src.models.projection_space_routing_hypergraph import ProjectionSpaceRoutingHyperNet

DATA_DIR = ROOT / 'data' / 'camels_us'
LOOKBACK, HIDDEN_DIM = 26, 64


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--device', default='cpu')
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--result-dir', type=Path, default=ROOT / 'results' / 'weekly_projection_space_routing')
    p.add_argument('--max-train-basins', type=int, default=128)
    p.add_argument('--use-orthogonal-init', action='store_true')
    p.add_argument('--nce-loss-weight', type=float, default=0.0)
    p.add_argument('--top-k', type=int, default=2)
    return p.parse_args()


def set_seed(seed):
    np.random.seed(seed); torch.manual_seed(seed)


def evaluate(model, loader, device):
    model.eval(); preds=[]; obs=[]; masks=[]
    with torch.no_grad():
        for forcing, static, target, mask in loader:
            pred, _ = model(forcing.to(device), static.to(device), mask.to(device))
            preds.append(pred.cpu()); obs.append(target); masks.append(mask)
    p = torch.cat(preds).numpy().ravel(); y = torch.cat(obs).numpy().ravel(); m = torch.cat(masks).numpy().ravel() > 0
    p, y = p[m], y[m]
    nse = 1 - np.sum((p-y)**2)/(np.sum((y-y.mean())**2)+1e-10)
    return {'nse': float(nse), 'rmse': float(np.sqrt(np.mean((p-y)**2))), 'corr': float(np.corrcoef(p, y)[0,1]), 'n': int(len(p))}


def main():
    args = parse_args(); set_seed(args.seed); args.result_dir.mkdir(parents=True, exist_ok=True)
    device = args.device if args.device in ['cpu','cuda','mps'] else 'cpu'
    if device == 'cuda' and not torch.cuda.is_available(): device = 'cpu'
    if device == 'mps' and not torch.backends.mps.is_available(): device = 'cpu'
    if device == 'cpu': torch.set_num_threads(min(16, os.cpu_count() or 8))
    data = load_camels_us(DATA_DIR)
    train = CAMELSSpatialDataset(data, LOOKBACK, 'train', max_basins=args.max_train_basins)
    val = CAMELSSpatialDataset(data, LOOKBACK, 'val', max_basins=data['N'])
    test = CAMELSSpatialDataset(data, LOOKBACK, 'test', max_basins=data['N'])
    train_loader = torch.utils.data.DataLoader(train, batch_size=args.batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val, batch_size=args.batch_size)
    test_loader = torch.utils.data.DataLoader(test, batch_size=args.batch_size)
    model = ProjectionSpaceRoutingHyperNet(args.max_train_basins, data['F'], data['D_s'], hidden_dim=HIDDEN_DIM, use_orthogonal_init=args.use_orthogonal_init, nce_loss_weight=args.nce_loss_weight, top_k=args.top_k).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    best, history, ckpt = -999, {'loss': [], 'val_nse': []}, args.result_dir / 'projection_space_routing_best.pt'
    aux_weights = {'graph_diversity_loss':0.01,'graph_coverage_loss':0.05,'structure_diversity_loss':0.005,'beta_mean':0.001,'nce_loss':args.nce_loss_weight}
    for ep in range(args.epochs):
        model.train(); total=0; nb=0; t0=time.time()
        for forcing, static, target, mask in train_loader:
            forcing, static, target, mask = forcing.to(device), static.to(device), target.to(device), mask.to(device)
            opt.zero_grad(); pred, aux = model(forcing, static, mask)
            loss = (((pred-target)**2)*mask).sum()/(mask.sum()+1e-8)
            for k,w in aux_weights.items():
                if k in aux: loss = loss + w*aux[k]
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            total += loss.item(); nb += 1
        val_metrics = evaluate(model, val_loader, device); history['loss'].append(total/max(nb,1)); history['val_nse'].append(val_metrics['nse'])
        if val_metrics['nse'] > best: best = val_metrics['nse']; torch.save(model.state_dict(), ckpt)
        if ep % 5 == 0 or ep == args.epochs-1: print(f"ep={ep} loss={total/max(nb,1):.4f} val_nse={val_metrics['nse']:.4f} time={time.time()-t0:.1f}s", flush=True)
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    test_metrics = evaluate(model, test_loader, device)
    out = {'projection_space_routing': {'params': sum(p.numel() for p in model.parameters()), 'best_val_nse': best, 'test': test_metrics, 'history': history}, 'config': vars(args) | {'device': device}}
    out['config']['result_dir'] = str(out['config']['result_dir'])
    with (args.result_dir/'results.json').open('w') as f: json.dump(out, f, indent=2)
    print(json.dumps(test_metrics, indent=2))

if __name__ == '__main__': main()
