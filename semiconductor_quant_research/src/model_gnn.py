import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from sklearn.preprocessing import RobustScaler
from scipy.stats import pearsonr, spearmanr
import warnings
warnings.filterwarnings('ignore')

FEATURE_COLS = [
    'mom_1d', 'mom_5d', 'mom_10d', 'mom_20d', 'mom_60d',
    'vol_5d', 'vol_20d', 'reversal_1d', 'vol_ratio',
    'dist_52w_high', 'cs_rank_mom10'
]
TARGET_COL  = 'fwd_ret_5d'
SEMI        = ['NVDA','AMD','AVGO','TSM','QCOM','AMAT',
               'LRCX','MU','KLAC','TXN','ASML','MRVL']
DEVICE      = torch.device('mps' if torch.backends.mps.is_available()
                            else 'cpu')
CORR_WIN    = 60
CORR_THRESH = 0.3


# ══════════════════════════════════════════════════════════════════
# ADJACENCY BUILDER
# ══════════════════════════════════════════════════════════════════

def adj_from_corr(ret:       pd.DataFrame,
                  tickers:   list,
                  date:      pd.Timestamp,
                  window:    int   = CORR_WIN,
                  threshold: float = CORR_THRESH):
    """
    Symmetric normalised adjacency  D^{-1/2} (A+I) D^{-1/2}
    using rolling correlation as edge weights.
    Returns None if not enough history.
    """
    idx = ret.index[ret.index <= date]
    if len(idx) < window:
        return None

    window_ret = ret[tickers].loc[idx[-window] : idx[-1]].dropna()
    if len(window_ret) < 10:
        return None

    corr = window_ret.corr().values.copy()
    n    = len(tickers)

    # Threshold + add self-loops
    A = np.where(corr > threshold, corr, 0.0).astype(np.float32)
    np.fill_diagonal(A, 0.0)
    A = A + np.eye(n, dtype=np.float32)

    # D^{-1/2} A D^{-1/2}
    d          = A.sum(axis=1)
    d_inv_sqrt = 1.0 / np.sqrt(np.maximum(d, 1e-8))
    A_norm     = (d_inv_sqrt[:, None] * A) * d_inv_sqrt[None, :]

    # Sanity check — reject if NaN survived
    if np.isnan(A_norm).any():
        return None

    return torch.FloatTensor(A_norm).to(DEVICE)


# ══════════════════════════════════════════════════════════════════
# MODEL
# ══════════════════════════════════════════════════════════════════

class GCNLayer(nn.Module):
    """
    Graph convolution: H' = σ(A_norm H W + b)
    Pure PyTorch — no torch_geometric dependency.
    """
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        nn.init.xavier_uniform_(self.linear.weight)

    def forward(self, x: torch.Tensor,
                adj: torch.Tensor) -> torch.Tensor:
        return F.elu(self.linear(adj @ x))


class SemiGNN(nn.Module):
    """
    2-layer GCN for semiconductor return prediction.
    Input : (n_stocks, n_features)
    Output: (n_stocks, 1)
    """
    def __init__(self, n_features: int = 11, hidden: int = 32):
        super().__init__()
        self.conv1 = GCNLayer(n_features, hidden)
        self.conv2 = GCNLayer(hidden, hidden)
        self.head  = nn.Linear(hidden, 1)
        self.drop  = nn.Dropout(0.1)
        self.norm1 = nn.LayerNorm(hidden)
        self.norm2 = nn.LayerNorm(hidden)

    def forward(self, x: torch.Tensor,
                adj: torch.Tensor) -> torch.Tensor:
        x = self.norm1(self.conv1(x, adj))
        x = self.drop(x)
        x = self.norm2(self.conv2(x, adj))
        return self.head(x)                  # (n_stocks, 1)


# ══════════════════════════════════════════════════════════════════
# DATA HELPER
# ══════════════════════════════════════════════════════════════════

def get_day_tensors(panel:   pd.DataFrame,
                    tickers: list,
                    date:    pd.Timestamp,
                    scaler:  RobustScaler):
    """
    Build node feature matrix X and target vector y for one day.
    Returns (None, None) on any NaN or missing data.
    """
    feats, targets = [], []
    for t in tickers:
        try:
            row = panel.xs(t, level='ticker').loc[date]
        except KeyError:
            return None, None

        f = row[FEATURE_COLS].values.astype(np.float32)
        y = float(row[TARGET_COL])

        if np.isnan(f).any() or np.isnan(y):   # ← key fix
            return None, None

        f_scaled = scaler.transform(f.reshape(1, -1))[0]
        feats.append(f_scaled)
        targets.append(y)

    X = torch.FloatTensor(np.array(feats)).to(DEVICE)   # (n_stocks, n_feat)
    y = torch.FloatTensor(np.array(targets)).unsqueeze(1).to(DEVICE)
    return X, y


# ══════════════════════════════════════════════════════════════════
# TRAINING + EVALUATION
# ══════════════════════════════════════════════════════════════════

def run_gnn(panel:  pd.DataFrame,
            ret:    pd.DataFrame,
            epochs: int   = 40,
            lr:     float = 5e-4) -> dict:

    print(f"\nRunning GNN on {DEVICE}...")
    print(f"  Corr window={CORR_WIN}d | threshold={CORR_THRESH}")

    tickers   = [t for t in SEMI
                 if t in panel.index.get_level_values('ticker').unique()]
    n_tickers = len(tickers)
    print(f"  Active tickers: {n_tickers}")

    dates       = panel.index.get_level_values('date').unique().sort_values()
    n           = len(dates)
    train_dates = dates[:int(n * 0.60)]
    test_dates  = dates[int(n * 0.60):]

    print(f"  Train: {train_dates[0].date()} → {train_dates[-1].date()}")
    print(f"  Test : {test_dates[0].date()}  → {test_dates[-1].date()}")

    # ── Fit scaler on training data ──
    train_panel = panel.loc[
        panel.index.get_level_values('date').isin(train_dates),
        FEATURE_COLS
    ].dropna()
    scaler = RobustScaler().fit(train_panel.values)

    # ── Model ──
    model     = SemiGNN(n_features=len(FEATURE_COLS), hidden=32).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(),
                                  lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=15, gamma=0.5)
    criterion = nn.MSELoss()

    print(f"  Model params: {sum(p.numel() for p in model.parameters()):,}")

    # ── Initialise best_state with current weights (fix NoneType error) ──
    best_loss  = np.inf
    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # ── Training ──
    for epoch in range(1, epochs + 1):
        model.train()
        ep_loss = 0.0
        n_days  = 0

        for date in train_dates:
            adj = adj_from_corr(ret, tickers, date)
            if adj is None:
                continue

            X, y = get_day_tensors(panel, tickers, date, scaler)
            if X is None:
                continue

            optimizer.zero_grad()
            pred = model(X, adj)
            loss = criterion(pred, y)

            # ── Skip NaN loss — never update on bad batch ──
            if torch.isnan(loss) or torch.isinf(loss):
                continue

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

            ep_loss += loss.item()
            n_days  += 1

        scheduler.step()
        avg_loss = ep_loss / max(n_days, 1)

        if avg_loss < best_loss and n_days > 0:
            best_loss  = avg_loss
            best_state = {k: v.cpu().clone()
                          for k, v in model.state_dict().items()}

        if epoch % 10 == 0:
            print(f"  Epoch {epoch:>3}/{epochs} | "
                  f"loss={avg_loss:.6f} | days={n_days}")

    # ── Restore best weights ──
    model.load_state_dict(best_state)
    model.eval()

    Path("models").mkdir(exist_ok=True)
    torch.save(best_state, "models/gnn.pt")
    print("  ✓ Saved models/gnn.pt")

    # ── Evaluation ──
    all_pred, all_true = {}, {}
    with torch.no_grad():
        for date in test_dates:
            adj = adj_from_corr(ret, tickers, date)
            if adj is None:
                continue
            X, y = get_day_tensors(panel, tickers, date, scaler)
            if X is None:
                continue
            pred = model(X, adj).cpu().numpy().flatten()
            true = y.cpu().numpy().flatten()
            for t, p, tr in zip(tickers, pred, true):
                d_str = str(date)[:10]
                if d_str not in all_pred:
                    all_pred[d_str] = {}
                    all_true[d_str] = {}
                all_pred[d_str][t] = float(p)
                all_true[d_str][t] = float(tr)

    pred_df         = pd.DataFrame(all_pred).T
    pred_df.index   = pd.to_datetime(pred_df.index)
    pred_df         = pred_df.sort_index()
    target_df       = pd.DataFrame(all_true).T
    target_df.index = pd.to_datetime(target_df.index)
    target_df       = target_df.sort_index()

    # ── IC ──
    ics, rank_ics = {}, {}
    for date in pred_df.index:
        if date not in target_df.index:
            continue
        p    = pred_df.loc[date].dropna()
        t    = target_df.loc[date].reindex(p.index).dropna()
        both = pd.concat([p, t], axis=1).dropna()
        if len(both) < 4:
            continue
        ics[date]      = pearsonr(both.iloc[:, 0], both.iloc[:, 1])[0]
        rank_ics[date] = spearmanr(both.iloc[:, 0], both.iloc[:, 1])[0]

    ic_series      = pd.Series(ics).sort_index()
    rank_ic_series = pd.Series(rank_ics).sort_index()
    icir  = ic_series.mean() / ic_series.std() if ic_series.std() > 0 else np.nan
    ricir = rank_ic_series.mean() / rank_ic_series.std() \
            if rank_ic_series.std() > 0 else np.nan

    print(f"\n{'='*55}")
    print(f"  IC Report — GNN")
    print(f"{'='*55}")
    print(f"  IC mean     : {ic_series.mean():.5f}")
    print(f"  IC std      : {ic_series.std():.5f}")
    print(f"  ICIR        : {icir:.4f}")
    print(f"  RankIC mean : {rank_ic_series.mean():.5f}")
    print(f"  RankICIR    : {ricir:.4f}")
    print(f"  IC pos%     : {(ic_series > 0).mean()*100:.1f}%")
    print(f"  N days      : {len(ic_series)}")
    print(f"{'='*55}")

    ic_series.to_csv("results/gnn_ic.csv",           header=['ic'])
    rank_ic_series.to_csv("results/gnn_rank_ic.csv", header=['rank_ic'])
    print("  ✓ Saved results/gnn_ic.csv")

    return {
        'model':          model,
        'ic_series':      ic_series,
        'rank_ic_series': rank_ic_series,
        'metrics': {
            'IC': ic_series.mean(), 'ICIR': icir,
            'RankIC': rank_ic_series.mean(), 'RankICIR': ricir,
            'IC_pos_pct': (ic_series > 0).mean(), 'N': len(ic_series)
        }
    }


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from src.features import load_features
    from src.data_loader import load

    panel = load_features()
    _, _, ret = load()

    print(f"Device: {DEVICE}")
    results = run_gnn(panel, ret, epochs=40)
