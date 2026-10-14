"""Rebuild every table and in-text number of the paper from results/*.csv.

    python tools/make_tables.py            -> prints, and writes results/TABLES.md

Run the experiment scripts first (or use the reference CSVs shipped in results/).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from clab import config as C  # noqa: E402
from clab.data import RESULTS  # noqa: E402
from clab.stats import holm_intervals, paired_ci  # noqa: E402

CLIP, RC = C.HEADLINE, C.DIAGNOSTIC
ORDER = C.ORDER
L = []
p = lambda s="": L.append(s)
f2 = lambda x: f"{x:.2f}"
sg = lambda x: f"{x:+.2f}"


def have(name):
    ok = (RESULTS / name).exists()
    if not ok:
        p(f"_({name} not found; run the corresponding experiment)_\n")
    return ok


def main_grid():
    G = pd.read_csv(RESULTS / "main_grid.csv")
    P = G.pivot_table(index=["setting", "backbone", "seed"], columns="method",
                      values="mAP").reset_index()
    return lambda sn, m, bb=CLIP: P[(P.setting == sn) & (P.backbone == bb)].sort_values("seed")[m].values


def t1():
    p("## Table 1. Datasets\n")
    if not have("datasets.csv"):
        return
    d = pd.read_csv(RESULTS / "datasets.csv")
    p("| dataset | images | cat. | labels/img | shared-6 images | removed |")
    p("|---|---|---|---|---|---|")
    for _, r in d.iterrows():
        p(f"| {r.dataset} | {r.images} | {r.categories} | {r.labels_per_image:.2f} | "
          f"{r.shared6_images} | {r.removed_pct:.1f}% ({r.removed}) |")
    p()


def t2_t3_t5(v):
    p("## Table 2. Main results, CLIP ViT-B/32 (mAP, mean +- SD over 20 seeds)\n")
    p("| method | " + " | ".join(ORDER) + " | mean |"); p("|---" * 6 + "|")
    for m in ["zeroshot_raw", "zeroshot", "probe", "clab_raw", "clab_long", "clab_a02",
              "clab_a04", "clab"]:
        cells = [f"{v(s, m).mean():.2f} +- {v(s, m).std(ddof=1):.2f}" for s in ORDER]
        p(f"| {m} | " + " | ".join(cells) + f" | {np.mean([v(s, m).mean() for s in ORDER]):.2f} |")
    p()
    p("## Table 3. CLAB minus probe, paired over 20 seeds, Holm over 8 contrasts\n")
    keys = [(s, b) for b in (CLIP, RC) for s in ORDER]
    res = holm_intervals([v(s, "clab", b) - v(s, "probe", b) for s, b in keys])
    p("| backbone | transfer | delta | 95% CI | Holm-adjusted CI | resolved |")
    p("|---|---|---|---|---|---|")
    for (s, b), (m, lo, hi, hl, hh, pv, lev) in zip(keys, res):
        p(f"| {b} | {s} | {sg(m)} | [{sg(lo)}, {sg(hi)}] | [{sg(hl)}, {sg(hh)}] | "
          f"{'yes' if hl > 0 or hh < 0 else 'no'} |")
    p()
    p("## Table 5. Ablation (mean delta mAP vs full CLAB over 4 transfers x 2 backbones)\n")
    arms = [(s, b) for s in ORDER for b in (CLIP, RC)]
    for m, lab in [("clab_raw", "raw anchors (no centering)"), ("zeroshot", "no source head"),
                   ("probe", "no semantic branch"), ("clab_long", "800-epoch head")]:
        p(f"- {lab}: {sg(np.mean([v(s, m, b).mean() - v(s, 'clab', b).mean() for s, b in arms]))}")
    g = [(b, s, v(s, "zeroshot", b).mean() - v(s, "zeroshot_raw", b).mean())
         for b in (CLIP, RC) for s in ORDER[:3]]
    p("\nCentering gain in zero-shot (targets UCM, AID, DFC15): " +
      ", ".join(f"{b.split('_')[0]} {s.split('->')[1]} {sg(x)}" for b, s, x in g))
    p()


def t4(v):
    p("## Table 4. Oracle-tuned CLIP adaptation baselines (CLIP ViT-B/32)\n")
    if not have("adapters20.csv"):
        return
    A = pd.read_csv(RESULTS / "adapters20.csv")
    g = A.groupby(["setting", "config"]).mAP.mean()
    p("| transfer | best baseline (config) | baseline | CLAB | delta [95% CI] |")
    p("|---|---|---|---|---|")
    for s in ORDER:
        best = g.loc[s].idxmax()
        q = A[(A.setting == s) & (A.config == best)].sort_values("seed").mAP.values
        m, lo, hi = paired_ci(v(s, "clab") - q)
        p(f"| {s} | {best} | {q.mean():.2f} | {v(s, 'clab').mean():.2f} | "
          f"{sg(m)} [{sg(lo)}, {sg(hi)}] |")
    tp = A.groupby(["setting", "family", "anchors"]).mAP.mean()
    t = [tp[(s, "tip_adapter", "centred")] - tp[(s, "tip_adapter", "raw")] for s in ORDER]
    c = [tp[(s, "clip_adapter", "centred")] - tp[(s, "clip_adapter", "raw")] for s in ORDER]
    p(f"\nCentering gain inside Tip-Adapter: {min(t):.2f} to {max(t):.2f}; "
      f"inside CLIP-Adapter: {min(c):.2f} to {max(c):.2f}\n")


def t6_t7():
    if not have("loto_selected.csv"):
        return
    p("## Table 6. Leave-one-transfer-out selection\n")
    S = pd.read_csv(RESULTS / "loto_selected.csv")
    p("| held out | alpha | epochs | CLAB | probe | delta [95% CI] |"); p("|---|---|---|---|---|---|")
    for _, r in S.iterrows():
        p(f"| {r.held_out} | {r.alpha:.2f} | {r.epochs} | {r.clab:.2f} | {r.probe:.2f} | "
          f"{sg(r.delta)} [{sg(r.lo)}, {sg(r.hi)}] |")
    p("\n## Table 7. Sensitivity to alpha (50 epochs) and to the budget (alpha = 0.30)\n")
    R = pd.read_csv(RESULTS / "loto_raw.csv"); R50 = R[R.epochs == 50]
    p("| alpha | " + " | ".join(ORDER) + " | mean |"); p("|---" * 6 + "|")
    for a in sorted(R50.alpha.unique()):
        q = R50[np.isclose(R50.alpha, a)]
        vals = [q[q.setting == s].mAP.mean() for s in ORDER]
        p(f"| {a:.2f} | " + " | ".join(f2(x) for x in vals) + f" | {np.mean(vals):.2f} |")
    for ep in [20, 50, 100, 200]:
        q = R[(R.epochs == ep) & np.isclose(R.alpha, 0.3)]
        p(f"- budget {ep}: mean {np.mean([q[q.setting == s].mAP.mean() for s in ORDER]):.2f}")
    p()


def t8():
    if not have("robust_mismatch.csv"):
        return
    p("## Table 8. Out-of-vocabulary imagery restored in the target (alpha = 0.30)\n")
    ms = pd.read_csv(RESULTS / "robust_mismatch.csv")
    p("| transfer | OOV % | probe | CLAB | delta [95% CI] |"); p("|---|---|---|---|---|")
    ds = []
    for s in ms.setting.unique():
        q = ms[ms.setting == s].sort_values("seed")
        m, lo, hi = paired_ci(q["u0.3"].values - q.probe.values); ds.append(m)
        p(f"| {s} | {100 * q.oov.mean():.1f} | {q.probe.mean():.2f} | {q['u0.3'].mean():.2f} | "
          f"{sg(m)} [{sg(lo)}, {sg(hi)}] |")
    worst = lambda col: min((ms[ms.setting == s][col] - ms[ms.setting == s].probe).mean()
                            for s in ms.setting.unique())
    p(f"\nMean over six: {sg(np.mean(ds))}; worst direction at alpha 0.20 / 0.30 / 0.40: "
      f"{sg(worst('u0.2'))} / {sg(worst('u0.3'))} / {sg(worst('u0.4'))}")
    if have("robust_standard.csv"):
        st = pd.read_csv(RESULTS / "robust_standard.csv")
        mean_gain = lambda a: np.mean([(st[st.setting == s][f"u{a}"] - st[st.setting == s].probe).mean()
                                       for s in ORDER])
        g = [mean_gain(a) for a in (0.2, 0.25, 0.3, 0.35, 0.4)]
        p(f"Standard transfers: mean gain over the probe varies by {max(g) - min(g):.2f} "
          f"for alpha in [0.20, 0.40]")
    p()


def t9(v):
    p("## Table 9. Published results (reproduced from the literature) and margins\n")
    f = ROOT / "supplementary" / "published_results.csv"
    if not f.exists():
        return
    P = pd.read_csv(f)
    best = {s: P[s].max() for s in ORDER}
    p("| method | backbone | " + " | ".join(ORDER) + " | source |"); p("|---" * 7 + "|")
    for _, r in P.iterrows():
        p(f"| {r.method} | {r.backbone} | " + " | ".join(f"{r[s]:.1f}" for s in ORDER) +
          f" | {r.source} |")
    p("| CLAB (ours) | CLIP ViT-B/32 | " + " | ".join(f"{v(s, 'clab').mean():.1f}" for s in ORDER) + " | |")
    p("\nMargin over best published: " +
      ", ".join(f"{s} {v(s, 'clab').mean() - best[s]:+.1f}" for s in ORDER))
    spread = {s: P[s].max() - P[s].min() for s in ORDER}
    p("Spread among published methods per transfer: " +
      ", ".join(f"{s} {x:.1f}" for s, x in spread.items()) + "\n")


def t10_t16(v):
    if have("backbone_study.csv"):
        p("## Table 10. Frozen-feature backbone study (5 head seeds, split seed 0)\n")
        B = pd.read_csv(RESULTS / "backbone_study.csv")
        g = B.groupby(["backbone", "setting"]).mAP.agg(["mean", "std"])
        p("| backbone | " + " | ".join(ORDER) + " |"); p("|---" * 5 + "|")
        for bb in B.backbone.unique():
            p(f"| {bb} | " + " | ".join(f"{g.loc[(bb, s), 'mean']:.2f} +- {g.loc[(bb, s), 'std']:.2f}"
                                        for s in ORDER) + " |")
        m = lambda bb, s: g.loc[(bb, s), "mean"]
        for bb in ["clip_vitb32_control", "clip_rn50_control"]:
            p(f"- {bb} minus ImageNet-1K TResNet-M: " +
              ", ".join(sg(m(bb, s) - m("imagenet_tresnet_m", s)) for s in ORDER))
        p("\n## Table 16. Contamination audit (RemoteCLIP ViT-B/32 minus CLIP ViT-B/32)\n")
        zs = lambda s, bb: v(s, "zeroshot", bb).mean()
        for tgt, s, s2 in [("UC Merced", "AID->UCM", None), ("AID", "UCM->AID", None),
                           ("DFC15", "AID->DFC15", "UCM->DFC15")]:
            fp = f"{sg(m('remoteclip_vitb32', s) - m('clip_vitb32_control', s))}"
            if s2:
                fp += f" / {sg(m('remoteclip_vitb32', s2) - m('clip_vitb32_control', s2))}"
            p(f"- {tgt}: zero-shot {sg(zs(s, RC) - zs(s, CLIP))}, frozen probe {fp}")
        p()


def t11_t12():
    if have("geometry_spectrum.csv"):
        p("## Table 11. Anchor geometry (CLIP ViT-B/32, 17 labels)\n")
        p(pd.read_csv(RESULTS / "geometry_spectrum.csv").round(1).to_markdown(index=False)
          if hasattr(pd.DataFrame, "to_markdown") else
          pd.read_csv(RESULTS / "geometry_spectrum.csv").round(1).to_string(index=False))
        p()
        p(pd.read_csv(RESULTS / "geometry_cosines.csv").round(3).to_string(index=False))
        p()
        p(pd.read_csv(RESULTS / "geometry_centering_set.csv").round(2).to_string(index=False))
        p()
    if have("prompt_ablation.csv"):
        p("## Table 12. Prompt families (mean over the three target datasets)\n")
        d = pd.read_csv(RESULTS / "prompt_ablation.csv")
        cr = "mAP_centered" if "mAP_centered" in d else "mAP_centred"
        s = d.groupby("prompt_set")[["cos_raw_mean", "mAP_raw", cr]].mean()
        s["gain"] = s[cr] - s.mAP_raw
        p(s.sort_values("cos_raw_mean").round(3).to_string())
        p(f"\nRange of raw mAP {s.mAP_raw.max() - s.mAP_raw.min():.2f}; "
          f"of centered mAP {s[cr].max() - s[cr].min():.2f}\n")


def t13():
    if not have("fusion.csv"):
        return
    p("## Table 13. Fusion rule (alpha = 0.30, CLIP ViT-B/32, 20 seeds)\n")
    d = pd.read_csv(RESULTS / "fusion.csv")
    g = d.groupby(["rule", "setting"]).mAP.mean().unstack()[ORDER]
    g["mean"] = g.mean(1)
    p(g.sort_values("mean", ascending=False).round(2).to_string())
    p(f"\nCost of no standardization: {g.loc['zscore', 'mean'] - g.loc['raw', 'mean']:.2f}\n")


def t14_prop1():
    if have("perclass.csv"):
        p("## Table 14. Per-class AP on the DFC15 transfers\n")
        p(pd.read_csv(RESULTS / "perclass.csv").round(2).to_string(index=False)); p()
    if have("invariance.csv"):
        IV = pd.read_csv(RESULTS / "invariance.csv").groupby("setting").mean(numeric_only=True)
        p("## Proposition 1 measurements\n")
        p(f"- rho on source: {IV.rho_src.min():.1f} to {IV.rho_src.max():.1f}; "
          f"on target: {IV.rho_tgt.min():.1f} to {IV.rho_tgt.max():.1f}")
        dt = (IV.target - IV.source)
        p(f"- target constants instead of source: mean {sg(dt.mean())}, worst {sg(dt.min())}")
        p(f"- zero locations: largest |change| {(IV.zero_means - IV.source).abs().max():.6f}\n")


def t15():
    if have("source_budget.csv"):
        p("## Source-head budget (Section 6.6, 5 seeds)\n")
        S = pd.read_csv(RESULTS / "source_budget.csv")
        g = S.groupby(["setting", "backbone", "epochs"])[["src_train_mAP", "tgt_linear_mAP"]].mean()
        p(g.round(2).to_string()); p()
    if have("alpha_oracle.csv"):
        O = pd.read_csv(RESULTS / "alpha_oracle.csv").groupby("setting").mean(numeric_only=True)
        p(f"Oracle gains over alpha = 0.30: global {(O.global_oracle - O.fixed).mean():.2f}, "
          f"per-class {(O.perclass_oracle - O.fixed).mean():.2f}\n")
    if have("alpha_estimators.csv"):
        p("## Table 15. Label-free per-class estimators vs fixed alpha = 0.30\n")
        E = pd.read_csv(RESULTS / "alpha_estimators.csv")
        ests = ["rank_agreement", "source_validation", "shrunk_source_validation", "self_training"]
        per = E.groupby(["setting", "backbone"])[["fixed"] + ests].mean()
        p("| estimator | mean delta | arms improved (of 8) | worst arm |"); p("|---|---|---|---|")
        for e in ests:
            d = per[e] - per["fixed"]
            p(f"| {e} | {sg(d.mean())} | {int((d > 0).sum())} | {sg(d.min())} |")
        p()


if __name__ == "__main__":
    v = main_grid()
    t1(); t2_t3_t5(v); t4(v); t6_t7(); t8(); t9(v); t10_t16(v); t11_t12(); t13(); t14_prop1(); t15()
    txt = "\n".join(L)
    (RESULTS / "TABLES.md").write_text(txt, encoding="utf-8")
    print(txt)
