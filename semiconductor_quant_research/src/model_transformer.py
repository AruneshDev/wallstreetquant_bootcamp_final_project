import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
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
TARGET_COL = 'fwd_ret_1d'
SEQ_LEN    = 20
DEVICE     = torch.device('mps' if torch.backends.mps.is_available()
                           else 'cpu')


# ══════════════════════════════════════════════════════════════════
# DATASET
# ══════════════════════════════════════════════════════════════════

class StockSeqDataset(Dataset):
    """
    Builds (seq_len × n_features) sequences per (ticker, date).
    Stores date and ticker as plain strings so PyTorch
    default_collate can handle them without errors.

    X shape : (seq_len, n_features)
    y shape : (1,)
    date    : str  e.g. '2025-06-01'
    ticker  : str  e.g. 'NVDA'
    """
    def __init__(self,
                 panel:      pd.DataFrame,
                 date_range: pd.Index,
                 scaler:     RobustScaler = None,
                 fit_scaler: bool = False):
        self.samples = []

        tickers = panel.index.get_level_values('ticker').unique()

        # Fit scaler on training data only
        if fit_scaler and scaler is not None:
            flat = panel.loc[
                panel.index.get_level_values('date').isin(date_range),
                FEATURE_COLS
            ].dropna()
            scaler.fit(flat.values)

        for ticker in tickers:
            try:
                ts = panel.xs(ticker, level='ticker')[
                    FEATURE_COLS + [TARGET_COL]
                ].sort_index()
            except KeyError:
                continue

            ts_dates = ts.index[ts.index.isin(date_range)]
            if len(ts_dates) < SEQ_LEN + 1:
                continue

            feat_arr = ts[FEATURE_COLS].values.copy()
            tgt_arr  = ts[TARGET_COL].values.copy()

            if scaler is not None:
                feat_arr = scaler.transform(feat_arr)

            date_list = ts.index.tolist()

            for i in range(SEQ_LEN, len(date_list)):
                d = date_list[i]
                if d not in date_range:
                    continue
                seq = feat_arr[i - SEQ_LEN : i]   # (SEQ_LEN, n_feat)
                tgt = tgt_arr[i]
                if np.isnan(seq).any() or np.isnan(tgt):
                    continue

                self.samples.append((
                    torch.FloatTensor(seq),
                    torch.FloatTensor([tgt]),
                    str(d)[:10],          # ← plain string 'YYYY-MM-DD'
                    str(ticker)           # ← plain string
                ))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        return self.samples[i]


# ══════════════════════════════════════════════════════════════════
# MODEL
# ══════════════════════════════════════════════════════════════════

class StockTransformer(nn.Module):
    """
    Transformer encoder for next-day return prediction.
    Input : (batch, seq_len=20, n_features=11)
    Output: (batch, 1)

    Architecture:
      Linear projection → learnable positional encoding
      → 2-layer TransformerEncoder (pre-norm)
      → take last timestep → LayerNorm → Linear head
    """
    def __init__(self,
                 n_features: int = 11,
                 d_model:    int = 32,
                 nhead:      int = 4,
                 num_layers: int = 2,
                 dropout:    float = 0.1):
        super().__init__()
        self.input_proj = nn.Linear(n_features, d_model)
        self.pos_embed  = nn.Embedding(100, d_model)
        encoder_layer   = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True          # pre-norm = more stable training
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T   = x.size(1)
        pos = torch.arange(T, device=x.device)
        x   = self.input_proj(x) + self.pos_embed(pos)  # (B, T, d_model)
        x   = self.encoder(x)                            # (B, T, d_model)
        return self.head(x[:, -1, :])                    # (B, 1)


# ══════════════════════════════════════════════════════════════════
# TRAINING HELPERS
# ══════════════════════════════════════════════════════════════════

def train_epoch(model, loader, optimizer,
                criterion, device) -> float:
    model.train()
    total_loss = 0.0
    for seq, tgt, _, _ in loader:
        seq = seq.to(device)
        tgt = tgt.to(device)
        optimizer.zero_grad()
        pred = model(seq)
        loss = criterion(pred, tgt)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / max(len(loader), 1)


def eval_ic(model, loader, device) -> tuple:
    """
    Run inference on loader, return (pred_df, target_df).
    Both DataFrames indexed by date, columns = tickers.
    """
    model.eval()
    all_pred, all_true = {}, {}

    with torch.no_grad():
        for seq, tgt, dates_b, tickers_b in loader:
            seq  = seq.to(device)
            pred = model(seq).cpu().numpy().flatten()
            true = tgt.numpy().flatten()
            for d, t, p, tr in zip(dates_b, tickers_b, pred, true):
                if d not in all_pred:
                    all_pred[d] = {}
                    all_true[d] = {}
                all_pred[d][t] = float(p)
                all_true[d][t] = float(tr)

    pred_df   = pd.DataFrame(all_pred).T
    pred_df.index = pd.to_datetime(pred_df.index)
    pred_df   = pred_df.sort_index()

    target_df = pd.DataFrame(all_true).T
    target_df.index = pd.to_datetime(target_df.index)
    target_df = target_df.sort_index()

    return pred_df, target_df


def compute_ic_series(pred_df: pd.DataFrame,
                       target_df: pd.DataFrame) -> tuple:
    """Return (ic_series, rank_ic_series) as daily pd.Series."""
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

    return (pd.Series(ics).sort_index(),
            pd.Series(rank_ics).sort_index())


def print_ic_report(ic: pd.Series,
                     rank_ic: pd.Series,
                     label: str) -> dict:
    icir  = ic.mean()      / ic.std()      if ic.std()      > 0 else np.nan
    ricir = rank_ic.mean() / rank_ic.std() if rank_ic.std() > 0 else np.nan
    print(f"\n{'='*55}")
    print(f"  IC Report — {label}")
    print(f"{'='*55}")
    print(f"  IC mean     : {ic.mean():.5f}")
    print(f"  IC std      : {ic.std():.5f}")
    print(f"  ICIR        : {icir:.4f}")
    print(f"  RankIC mean : {rank_ic.mean():.5f}")
    print(f"  RankICIR    : {ricir:.4f}")
    print(f"  IC pos%     : {(ic > 0).mean()*100:.1f}%")
    print(f"  N days      : {len(ic)}")
    print(f"{'='*55}")
    return {
        'IC': ic.mean(), 'ICIR': icir,
        'RankIC': rank_ic.mean(), 'RankICIR': ricir,
        'IC_pos_pct': (ic > 0).mean(), 'N': len(ic)
    }


# ══════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ══════════════════════════════════════════════════════════════════

def run_transformer(panel:      pd.DataFrame,
                    epochs:     int   = 30,
                    batch_size: int   = 64,
                    lr:         float = 3e-4) -> dict:

    print(f"\nRunning Transformer on {DEVICE}...")
    print(f"  seq_len={SEQ_LEN} | epochs={epochs} | "
          f"batch={batch_size} | lr={lr}")

    dates = panel.index.get_level_values('date').unique().sort_values()
    n     = len(dates)

    # 60 / 40 split — enough for 2-year dataset
    split        = int(n * 0.60)
    train_dates  = dates[:split]
    test_dates   = dates[split:]

    print(f"  Train: {train_dates[0].date()} → "
          f"{train_dates[-1].date()} ({len(train_dates)}d)")
    print(f"  Test : {test_dates[0].date()}  → "
          f"{test_dates[-1].date()}  ({len(test_dates)}d)")

    # ── Datasets ──
    scaler   = RobustScaler()
    train_ds = StockSeqDataset(panel, train_dates,
                                scaler=scaler, fit_scaler=True)
    test_ds  = StockSeqDataset(panel, test_dates,
                                scaler=scaler, fit_scaler=False)

    print(f"  Train samples: {len(train_ds)} | "
          f"Test samples: {len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True,  drop_last=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size,
                              shuffle=False, drop_last=False)

    # ── Model ──
    model     = StockTransformer(n_features=len(FEATURE_COLS)).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(),
                                   lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs)
    criterion = nn.MSELoss()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model params: {n_params:,}")

    # ── Training ──
    best_loss, best_state = np.inf, None

    for epoch in range(1, epochs + 1):
        tr_loss = train_epoch(model, train_loader,
                              optimizer, criterion, DEVICE)
        scheduler.step()

        if tr_loss < best_loss:
            best_loss  = tr_loss
            best_state = {k: v.cpu().clone()
                          for k, v in model.state_dict().items()}

        if epoch % 5 == 0:
            print(f"  Epoch {epoch:>3}/{epochs} | "
                  f"loss={tr_loss:.6f}")

    # ── Restore best weights ──
    model.load_state_dict(best_state)
    Path("models").mkdir(exist_ok=True)
    torch.save(best_state, "models/transformer.pt")
    print("  ✓ Saved models/transformer.pt")

    # ── Evaluate ──
    pred_df, target_df = eval_ic(model, test_loader, DEVICE)
    ic_series, rank_ic_series = compute_ic_series(pred_df, target_df)
    metrics = print_ic_report(ic_series, rank_ic_series, "Transformer")

    # ── Save ──
    ic_series.to_csv("results/transformer_ic.csv", header=['ic'])
    rank_ic_series.to_csv("results/transformer_rank_ic.csv",
                           header=['rank_ic'])
    print("  ✓ Saved results/transformer_ic.csv")

    return {
        'model':            model,
        'ic_series':        ic_series,
        'rank_ic_series':   rank_ic_series,
        'metrics':          metrics,
        'pred_df':          pred_df,
        'target_df':        target_df,
    }


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from src.features import load_features

    panel = load_features()
    print(f"Panel : {panel.shape}")
    print(f"Device: {DEVICE}")

    results = run_transformer(panel, epochs=30)
