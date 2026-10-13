"""Fine-tune a transformer on facts-only text, and see whether the 0.666 ceiling survives.

Readings were pre-registered in PLAN.md §2 before any run: <=0.69 strengthens the ceiling claim,
0.69-0.75 softens it, >=0.75 overturns it. The `last512` arm is a CAPABILITY CONTROL (§3) -- leaky
by construction, and its number is never reported as a result. Its job is to distinguish "the task
has no more signal" from "our fine-tuning is broken", which are otherwise indistinguishable.

Metrics are always **case-level**: chunk probabilities are averaged per case. Chunk-level accuracy is
printed too because it flatters, and seeing both prevents quoting the wrong one.

Model selection uses the **dev** split only. The test split is touched once, at the end.

Several seeds are run because a single fine-tuning run is a noisy estimate, and quoting one number
as *the* result would misrepresent it. The spread across seeds is reported with the mean.
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset

from src import paths as v1

CACHE = v1.ROOT / "ft" / "cache"
EXP = v1.ROOT / "ft" / "experiments"
EXP.mkdir(parents=True, exist_ok=True)


class Chunks(Dataset):
    def __init__(self, arm: str, part: str):
        self.rows = [json.loads(l) for l in open(CACHE / f"{arm}_{part}.jsonl")]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        return self.rows[i]


def collate(batch, pad_id: int):
    n = max(len(b["input_ids"]) for b in batch)
    ids = torch.full((len(batch), n), pad_id, dtype=torch.long)
    att = torch.zeros((len(batch), n), dtype=torch.long)
    for i, b in enumerate(batch):
        k = len(b["input_ids"])
        ids[i, :k] = torch.tensor(b["input_ids"], dtype=torch.long)
        att[i, :k] = 1
    return (ids, att,
            torch.tensor([b["label"] for b in batch], dtype=torch.long),
            [b["case_id"] for b in batch])


@torch.no_grad()
def evaluate(model, loader, device) -> dict:
    model.eval()
    probs, labels, cases = [], [], []
    for ids, att, y, cid in loader:
        out = model(input_ids=ids.to(device), attention_mask=att.to(device)).logits
        p = torch.softmax(out.float(), -1)[:, 1].cpu().numpy()
        probs.append(p)
        labels.append(y.numpy())
        cases.extend(cid)
    probs = np.concatenate(probs)
    labels = np.concatenate(labels)

    # chunk level -- printed because it flatters, never the headline
    chunk = {"auroc": float(roc_auc_score(labels, probs)),
             "acc": float(accuracy_score(labels, probs > 0.5))}

    # case level -- average the chunk probabilities per case
    agg: dict[str, list] = {}
    ylab: dict[str, int] = {}
    for c, p, y in zip(cases, probs, labels):
        agg.setdefault(c, []).append(p)
        ylab[c] = int(y)
    ids_ = sorted(agg)
    cp = np.array([np.mean(agg[c]) for c in ids_])
    cy = np.array([ylab[c] for c in ids_])
    case = {"auroc": float(roc_auc_score(cy, cp)),
            "acc": float(accuracy_score(cy, cp > 0.5)),
            "macro_f1": float(f1_score(cy, cp > 0.5, average="macro")),
            "n_cases": len(ids_)}
    return {"chunk": chunk, "case": case, "case_probs": {c: float(np.mean(agg[c])) for c in ids_}}


def run_seed(arm: str, model_name: str, seed: int, epochs: int, bs: int, lr: float,
             device: str, limit: int = 0) -> dict:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    torch.manual_seed(seed)
    np.random.seed(seed)
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=2).to(device)

    tr, dv, te = Chunks(arm, "train"), Chunks(arm, "dev"), Chunks(arm, "test")
    if limit:
        tr.rows = tr.rows[:limit]
        dv.rows = dv.rows[: max(200, limit // 5)]
    col = lambda b: collate(b, tok.pad_token_id)
    dl_tr = DataLoader(tr, batch_size=bs, shuffle=True, collate_fn=col)
    dl_dv = DataLoader(dv, batch_size=bs * 2, collate_fn=col)
    dl_te = DataLoader(te, batch_size=bs * 2, collate_fn=col)

    ytr = np.array([r["label"] for r in tr.rows])
    w = torch.tensor([len(ytr) / (2 * max(1, (ytr == 0).sum())),
                      len(ytr) / (2 * max(1, (ytr == 1).sum()))],
                     dtype=torch.float32).to(device)
    lossf = torch.nn.CrossEntropyLoss(weight=w)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total = epochs * len(dl_tr)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=total,
                                                pct_start=0.1)
    print(f"  seed {seed}: {len(tr):,} train chunks, {len(dl_tr)} steps/epoch x {epochs}",
          flush=True)

    best = {"case": {"auroc": 0.0}}
    best_state = None
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        run = 0.0
        for i, (ids, att, y, _) in enumerate(dl_tr, 1):
            opt.zero_grad()
            out = model(input_ids=ids.to(device), attention_mask=att.to(device)).logits
            loss = lossf(out, y.to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            run += loss.item()
            if i % 200 == 0:
                el = time.time() - t0
                print(f"    ep{ep+1} {i}/{len(dl_tr)}  loss {run/i:.4f}  "
                      f"({i/el:.1f} it/s)", flush=True)
        dev = evaluate(model, dl_dv, device)
        print(f"    ep{ep+1} DEV case AUROC {dev['case']['auroc']:.4f} "
              f"acc {dev['case']['acc']:.4f}  (chunk {dev['chunk']['auroc']:.4f})", flush=True)
        if dev["case"]["auroc"] > best["case"]["auroc"]:
            best = dev
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    test = evaluate(model, dl_te, device)
    print(f"  seed {seed} TEST case AUROC {test['case']['auroc']:.4f} "
          f"acc {test['case']['acc']:.4f} mF1 {test['case']['macro_f1']:.4f} "
          f"({(time.time()-t0)/60:.1f} min)", flush=True)
    return {"seed": seed, "dev_best": {k: best[k] for k in ("chunk", "case")},
            "test": {k: test[k] for k in ("chunk", "case")},
            "test_case_probs": test["case_probs"]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=("masked", "last512"), default="masked")
    ap.add_argument("--model", default="law-ai/InLegalBERT")
    ap.add_argument("--seeds", type=int, nargs="+", default=[573])
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--limit", type=int, default=0, help="smoke test: cap train chunks")
    args = ap.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"=== ft {args.arm} | {args.model} | device={device} ===")
    if args.arm == "last512":
        print("!! CAPABILITY CONTROL. This arm is leaky by construction (v1: ~85% of its")
        print("!! advantage is residual disposition language). Never report it as a result.\n")

    runs = [run_seed(args.arm, args.model, s, args.epochs, args.bs, args.lr, device, args.limit)
            for s in args.seeds]
    au = [r["test"]["case"]["auroc"] for r in runs]
    ac = [r["test"]["case"]["acc"] for r in runs]

    BAR_AUROC, BAR_ACC = 0.666, 0.621
    print(f"\n  case-level TEST across {len(runs)} seed(s):")
    print(f"    AUROC   mean {np.mean(au):.4f}  sd {np.std(au):.4f}  "
          f"({', '.join(f'{a:.3f}' for a in au)})")
    print(f"    acc     mean {np.mean(ac):.4f}  sd {np.std(ac):.4f}")
    print(f"    v1 TF-IDF bar on the same split: AUROC {BAR_AUROC:.3f}, acc {BAR_ACC:.3f}")

    verdict = None
    if args.limit:
        print("\n  !! --limit was set: this is a SMOKE TEST, not a result. No verdict is issued")
        print("  !! and nothing is written to experiments/.")
        return
    if args.arm == "masked":
        m = float(np.mean(au))
        verdict = ("CEILING STRENGTHENED: a domain-pretrained transformer, fine-tuned on all the "
                   "text, gets no further than word counts. The wall is the task, not the method."
                   if m <= 0.69 else
                   "CEILING SOFTENS: a real but modest gain. Report this as the new ceiling and "
                   "revise the estimate in report 15.6."
                   if m < 0.75 else
                   "CEILING CLAIM WRONG: the facts carry substantially more than this project could "
                   "extract. The diffuse-signal and 0.043-headroom conclusions must be re-examined.")
        print(f"\n  PRE-REGISTERED VERDICT: {verdict}")

    # per-case probabilities, kept OUT of the json but saved, so the fine-tuned arm can be
    # compared to v1's TF-IDF arm by paired bootstrap on identical cases. The first run omitted
    # these and no paired comparison was possible without retraining.
    import numpy as _np
    tag = f"{args.arm}_{args.model.split('/')[-1]}"
    for r in runs:
        cp = r["test_case_probs"]
        ids = sorted(cp)
        _np.savez_compressed(EXP / f"probs_{tag}_seed{r['seed']}.npz",
                             case_ids=_np.array(ids),
                             probs=_np.array([cp[c] for c in ids], dtype=_np.float32))
    out = EXP / f"ft_{tag}.json"
    out.write_text(json.dumps({
        "arm": args.arm, "model": args.model, "device": device,
        "epochs": args.epochs, "bs": args.bs, "lr": args.lr, "seeds": args.seeds,
        "is_capability_control": args.arm == "last512",
        "v1_bar": {"auroc": BAR_AUROC, "acc": BAR_ACC, "method": "TF-IDF + logistic regression"},
        "test_case_auroc_mean": round(float(np.mean(au)), 4),
        "test_case_auroc_sd": round(float(np.std(au)), 4),
        "test_case_acc_mean": round(float(np.mean(ac)), 4),
        "preregistered_verdict": verdict,
        "runs": [{k: v for k, v in r.items() if k != "test_case_probs"} for r in runs]},
        indent=1))
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
