# -*- coding: utf-8 -*-
"""
以 HyperGraphRAG 三疾病模板为准，对 COPD 三类 251 例（四维链可用子集）重跑检索式实验。

两种检索模式：
  1) pooled  : 全库 LOO —— 每个查询都在 chronic+ae+infection 合并候选库中检索（排除自身），
               公平检验三病种判别力（对应原实验 Task1 三分类 / Task2 恶化二分类 / Task3 严重度 kNN）。
  2) routing : 生产路由 —— 不指定 dataset_scope，由 resolve_auto_dataset_ids 按主诊断自动路由
               （稳定→chronic；急性加重→ae+infection+chronic；感染→infection+ae+chronic）。

健壮性：
  - 增量保存（每例落盘 JSONL + CSV），支持断点续跑（--resume 跳过已处理 case_id）
  - 单例超时保护（--timeout 秒，默认 240s；超时线程 daemon 丢弃，跳过该例）
  - 失败/空结果记录在 errors 列表

用法：
  cd <experiment_dir>
  python3 run_hgr_eval.py [--limit N] [--timeout 240] [--out 实验结果_HyperGraphRAG.json]
"""
import argparse
import json
import os
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

BASE_TR = Path(os.environ.get('TRAIN_ROOT', '/path/to/training_set')) / "出院主诊断_COPD三类"
OUT_DIR = Path(os.environ.get('EXPERIMENT_BASE', os.getcwd()))

GROUPS = [
    "慢性阻塞性肺病",
    "慢性阻塞性肺病伴有急性加重",
    "慢性阻塞性肺病伴有急性下呼吸道感染",
]
GROUP_LABEL = {
    "慢性阻塞性肺病": "A_stable",
    "慢性阻塞性肺病伴有急性加重": "B_ae",
    "慢性阻塞性肺病伴有急性下呼吸道感染": "C_infection",
}
TEMPLATE_BY_LABEL = {"A_stable": "copd_stable", "B_ae": "copd_ae", "C_infection": "copd_infection"}
TEMPLATE_TO_LABEL = {v: k for k, v in TEMPLATE_BY_LABEL.items()}
BUCKETS = {
    "history_chain": "history_background_similarity.json",
    "evidence_chain": "clinical_evidence_chain_similarity.json",
    "care_chain": "treatment_trajectory_similarity.json",
    "severity_chain": "clinical_severity_similarity.json",
}
POOL_DATASETS = ["chronic", "ae", "infection"]

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, os.environ.get('HYPERGRAPHRAG_PATH', str(_REPO / 'third_party' / 'HyperGraphRAG')))
from dashboard_case_app import DashboardStore  # noqa: E402


def load_cases() -> dict[str, dict]:
    """从训练集四维相似链输出加载全部链可用病例。"""
    cases: dict[str, dict] = {}
    for group in GROUPS:
        cdir = BASE_TR / group / "四维相似链输出"
        raw_by_bucket: dict[str, dict[str, list]] = {}
        for bucket, fname in BUCKETS.items():
            raw = json.load(open(cdir / fname, encoding="utf-8"))
            per: dict[str, list] = {}
            for c in raw["病例相似链数据"]:
                ident = c["病例标识"]
                groups = [
                    {
                        "g": x.get("字段分组", ""),
                        "s": x.get("相似链条", ""),
                        "d": x.get("提取结果", {}),
                    }
                    for x in c.get("字段分组明细", [])
                ]
                per[ident["病例编号"]] = groups
            raw_by_bucket[bucket] = per
        for cid in raw_by_bucket["history_chain"]:
            cases[cid] = {
                "group": group,
                "label": GROUP_LABEL[group],
                "ident": None,
                "groups": {b: raw_by_bucket[b][cid] for b in BUCKETS},
            }
        # 从 history 文件取病例标识（入院/出院时间用于住院天数）
        raw0 = json.load(open(cdir / BUCKETS["history_chain"], encoding="utf-8"))
        for c in raw0["病例相似链数据"]:
            i = c["病例标识"]
            if i["病例编号"] in cases:
                cases[i["病例编号"]]["ident"] = i
    return cases


def hospital_days(ident: dict) -> float | None:
    try:
        a = datetime.strptime(ident.get("入院时间", "").strip(), "%Y-%m-%d")
        d = datetime.strptime(ident.get("出院时间", "").strip(), "%Y-%m-%d")
        return max(0.0, float((d - a).days))
    except Exception:
        return None


def build_payload(case: dict) -> dict:
    payload = {"diagnosis": case["ident"]["主诊断"], "query_case_id": ""}
    for bucket in BUCKETS:
        payload[bucket] = ""
        payload[bucket.replace("_chain", "_groups")] = case["groups"][bucket]
    return payload


def summarize_ranked(items: list[dict]) -> dict:
    """把排好序的结果列表（已排除自身）压缩为 top1/top3 摘要。"""
    out = {
        "top1_case_id": "", "top1_tpl": "", "top1_ds": "",
        "top1_final": 0.0, "top1_sem": 0.0, "top1_str": 0.0,
        "top1_rule": 0.0, "top1_dis": 0.0, "top1_hier": 0.0,
        "top3_tpls": [], "n_candidates": len(items),
    }
    if not items:
        return out
    t1 = items[0]
    out.update({
        "top1_case_id": t1.get("case_id", ""),
        "top1_tpl": t1.get("template_hint", ""),
        "top1_ds": t1.get("dataset_id", ""),
        "top1_final": float(t1.get("final_score", 0) or 0),
        "top1_sem": float(t1.get("semantic_score", 0) or 0),
        "top1_str": float(t1.get("structured_score", 0) or 0),
        "top1_rule": float(t1.get("guideline_rule_score", 0) or 0),
        "top1_dis": float(t1.get("disease_consistency_score", 0) or 0),
        "top1_hier": float(t1.get("diagnosis_hierarchy_score", 0) or 0),
        "top3_tpls": [str(x.get("template_hint", "")) for x in items[:3]],
    })
    return out


def run_one(ds: DashboardStore, case: dict) -> dict:
    """执行 pooled + routing 两种检索，返回该例的完整结果。"""
    payload = build_payload(case)
    row = {
        "case_id": "", "group": case["group"], "label": case["label"],
        "diagnosis": case["ident"]["主诊断"], "hospital_days": hospital_days(case["ident"]),
        "pooled": None, "routing": None, "self_in_top": False,
    }
    case_id = case["ident"]["病例编号"] if case["ident"] else None
    row["case_id"] = case_id
    payload["query_case_id"] = case_id or ""

    # ---- pooled（全库 LOO）+ routing ----
    # B/C 查询自动路由已覆盖 chronic+ae+infection 三库，pooled 与 routing 同源，一次调用即可；
    # A(稳定) 自动路由仅 chronic，pooled 需显式逐库补齐全库候选，routing 保持生产路由（仅 chronic）。
    if case["group"] != "慢性阻塞性肺病":
        p = dict(payload)
        p["dataset_scope"] = "all"
        resp = ds.similar_cases(p, top_k=6)
        merged = list(resp.get("results", []) or [])
        routed = list(merged)
    else:
        merged = []
        chronic_res: list[dict] = []
        for dsid in POOL_DATASETS:
            p = dict(payload)
            p["dataset_scope"] = dsid
            resp = ds.similar_cases(p, top_k=6)
            res = list(resp.get("results", []) or [])
            merged.extend(res)
            if dsid == "chronic":
                chronic_res = res
        merged.sort(key=lambda x: x.get("final_score", 0.0), reverse=True)
        # 生产路由（稳定→chronic）即为 chronic 库检索结果
        routed = list(chronic_res)

    if case_id:
        before = len(merged)
        merged = [x for x in merged if x.get("case_id") != case_id]
        row["self_in_top"] = before > len(merged)
        routed = [x for x in routed if x.get("case_id") != case_id]
    row["pooled"] = summarize_ranked(merged)
    row["routing"] = summarize_ranked(routed)
    return row


def _run_one_safe(ds: DashboardStore, case: dict, timeout: float) -> dict | None:
    result: dict = {}
    error = {}

    def worker():
        try:
            result.update(run_one(ds, case))
        except Exception as e:  # noqa: BLE001
            error["msg"] = f"{type(e).__name__}: {e}"
            error["tb"] = traceback.format_exc()

    t = threading.Thread(target=worker, daemon=True)
    t.start()
    t.join(timeout=timeout)
    if t.is_alive():
        return {"_timeout": True, "case_id": case.get("_cid"), "error": f"timeout>{timeout}s"}
    if error:
        return {"_error": True, "case_id": case.get("_cid"), "error": error["msg"][:500]}
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--timeout", type=int, default=240)
    ap.add_argument("--out", default="实验结果_HyperGraphRAG.json")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    cases = load_cases()
    print(f"载入链可用病例: {len(cases)}", flush=True)
    counts = {}
    for c in cases.values():
        counts[c["label"]] = counts.get(c["label"], 0) + 1
    print("分组:", counts, flush=True)

    order = sorted(cases.items(), key=lambda kv: (kv[1]["label"], kv[0]))
    if args.limit > 0:
        order = order[: args.limit]

    jsonl_path = OUT_DIR / "hgr_eval_rows.jsonl"
    done_ids: set[str] = set()
    if args.resume and jsonl_path.exists():
        with open(jsonl_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    done_ids.add(json.loads(line).get("case_id", ""))
                except Exception:
                    pass
        print(f"断点续跑: 已处理 {len(done_ids)} 例，跳过", flush=True)

    ds = DashboardStore()
    print("DashboardStore 初始化完成", flush=True)

    rows = []
    t_start = time.perf_counter()
    for idx, (cid, case) in enumerate(order):
        if cid in done_ids:
            continue
        case["_cid"] = cid
        t0 = time.perf_counter()
        row = _run_one_safe(ds, case, args.timeout)
        dt = time.perf_counter() - t0
        if row is None:
            row = {"_error": True, "case_id": cid, "error": "no result"}
        if row.get("_timeout") or row.get("_error"):
            print(f"[{idx+1}/{len(order)}] {cid} ❌ {row.get('error','')[:80]} ({dt:.0f}s)", flush=True)
            with open(jsonl_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            continue
        row["elapsed_s"] = round(dt, 1)
        rows.append(row)
        with open(jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        done = len(done_ids) + len(rows)
        if (idx + 1) % 5 == 0 or idx == len(order) - 1:
            eta = (time.perf_counter() - t_start) / max(1, done) * max(1, len(order) - done)
            pooled_tpl = (row.get("pooled") or {}).get("top1_tpl", "")
            print(
                f"[{idx+1}/{len(order)}] {cid} {row['label']} "
                f"pooled_top1={pooled_tpl} final={((row.get('pooled') or {}).get('top1_final',0)):.3f} "
                f"({dt:.0f}s, ETA {eta/60:.1f}min)", flush=True
            )

    # ---- 汇总 ----
    rows = [r for r in rows if not r.get("_timeout") and not r.get("_error")]
    summary = aggregate(rows)
    summary["meta"] = {
        "query_total": len(cases),
        "evaluated": len(rows),
        "coverage_pct": round(len(rows) / 253 * 100, 1),
        "note": "251/253 例有四维相似链（AE 1 例、感染 1 例缺链无法构建查询表示）；253 为原超图实验总例数",
        "timeout_seconds": args.timeout,
        "run_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    out_path = OUT_DIR / args.out
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n✅ 汇总已写入 {out_path}")
    t1 = summary["task1_three_class"]
    print("── Task1 三分类 ──")
    print("  pooled : accuracy", t1["pooled"]["accuracy"], "| macro_f1", t1["pooled"]["macro_f1"])
    print("  routing: accuracy", t1["routing"]["accuracy"], "| macro_f1", t1["routing"]["macro_f1"])
    print("  top3 命中率:", t1["top3_hit_rate"])
    print("  per-class f1 (pooled):", {k: v["f1"] for k, v in t1["pooled"]["per_class_f1"].items()})
    print("── Task2 二分类 A vs B+C ──")
    print("  pooled :", summary["task2_binary_A_vs_BC"]["pooled"])
    print("  routing:", summary["task2_binary_A_vs_BC"]["routing"])
    print("  AUC(final/sem):", summary["task2_binary_A_vs_BC"]["auc_final_score"],
          summary["task2_binary_A_vs_BC"]["auc_semantic_score"])
    print("── Task3 严重度 kNN ──")
    print("  pooled :", summary["task3_severity_knn"]["pooled"])
    print("  routing:", summary["task3_severity_knn"]["routing"])


def aggregate(rows: list[dict]) -> dict:
    from sklearn.metrics import (
        accuracy_score, f1_score, precision_score, recall_score, confusion_matrix,
        roc_auc_score,
    )

    labels = ["A_stable", "B_ae", "C_infection"]
    gtruth = [r["label"] for r in rows]
    pred_pooled = []
    pred_routing = []
    for r in rows:
        pred_pooled.append(TEMPLATE_TO_LABEL.get((r.get("pooled") or {}).get("top1_tpl", ""), ""))
        pred_routing.append(TEMPLATE_TO_LABEL.get((r.get("routing") or {}).get("top1_tpl", ""), ""))
    gt_tpl = [TEMPLATE_BY_LABEL[l] for l in gtruth]

    def cls_stats(pred):
        # pred 为类别标签（A_stable/B_ae/C_infection）；空模板预测归入 unknown
        tpl_list = list(TEMPLATE_BY_LABEL.values())
        pred_norm = [TEMPLATE_BY_LABEL.get(p, "unknown") for p in pred]
        acc = accuracy_score(gt_tpl, pred_norm)
        f1m = f1_score(gt_tpl, pred_norm, average="macro", zero_division=0, labels=tpl_list)
        cm = confusion_matrix(gt_tpl, pred_norm, labels=tpl_list + ["unknown"]).tolist()
        per = {}
        for l in labels:
            y_true = [1 if t == TEMPLATE_BY_LABEL[l] else 0 for t in gt_tpl]
            y_pred = [1 if p == l else 0 for p in pred]
            per[l] = {
                "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
                "recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
                "f1": round(f1_score(y_true, y_pred, zero_division=0), 4),
                "support": y_true.count(1),
            }
        return {
            "accuracy": round(acc, 4),
            "macro_f1": round(f1m, 4),
            "per_class_f1": per,
            "confusion_matrix": cm,
            "unknown_predictions": pred_norm.count("unknown"),
        }

    # top3 命中（任一 top3 模板 == 真实模板）
    def top3_hit(mode):
        hits = 0
        for r in rows:
            tpls = (r.get(mode) or {}).get("top3_tpls", [])
            if TEMPLATE_BY_LABEL[r["label"]] in tpls:
                hits += 1
        return round(hits / len(rows), 4)

    # ---- Task2 二分类: A(0) vs B+C(1) ----
    bin_true = [1 if l != "A_stable" else 0 for l in gtruth]
    bin_pred_pooled = [1 if p in ("B_ae", "C_infection") else 0 for p in pred_pooled]
    bin_pred_routing = [1 if p in ("B_ae", "C_infection") else 0 for p in pred_routing]

    def bin_stats(pred):
        return {
            "accuracy": round(accuracy_score(bin_true, pred), 4),
            "precision": round(precision_score(bin_true, pred, zero_division=0), 4),
            "recall": round(recall_score(bin_true, pred, zero_division=0), 4),
            "f1": round(f1_score(bin_true, pred, zero_division=0), 4),
        }

    # AUC：用 top1 final_score（注意模板条件化天花板，附注）与 top1 semantic_score
    def auc_of(score_key, mode):
        scores = []
        valid_true = []
        for r in rows:
            v = (r.get(mode) or {}).get(score_key, 0.0)
            if v is None:
                v = 0.0
            scores.append(float(v))
            valid_true.append(1 if r["label"] != "A_stable" else 0)
        try:
            return round(roc_auc_score(valid_true, scores), 4)
        except Exception:
            return None

    # ---- Task3: 检索式 kNN 严重度（住院天数代理） ----
    def knn_reg(mode):
        y_true, y_pred = [], []
        for r in rows:
            tgt_case = (r.get(mode) or {}).get("top1_case_id", "")
            if not tgt_case or tgt_case not in CASE_META:
                continue
            if r.get("hospital_days") is None or CASE_META[tgt_case].get("days") is None:
                continue
            y_true.append(r["hospital_days"])
            y_pred.append(CASE_META[tgt_case]["days"])
        from sklearn.metrics import mean_absolute_error, mean_squared_error
        import numpy as np
        pearson = float(np.corrcoef(y_true, y_pred)[0, 1]) if len(y_true) > 1 else None
        return {
            "n": len(y_true),
            "pearson_r": round(pearson, 4) if pearson is not None else None,
            "mae": round(float(mean_absolute_error(y_true, y_pred)), 4),
            "rmse": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 4),
            "mean_true": round(float(np.mean(y_true)), 2),
            "mean_pred": round(float(np.mean(y_pred)), 2),
        }

    # 组间得分统计（pooled top1）
    score_keys = ["top1_final", "top1_sem", "top1_str", "top1_rule", "top1_dis", "top1_hier"]
    import numpy as np
    group_stats = {}
    for l in labels:
        sub = [r for r in rows if r["label"] == l]
        gs = {}
        for k in score_keys:
            vals = [float((r.get("pooled") or {}).get(k, 0.0) or 0.0) for r in sub]
            gs[k] = {"mean": round(float(np.mean(vals)), 4), "std": round(float(np.std(vals)), 4)}
        gs["n"] = len(sub)
        group_stats[l] = gs

    return {
        "task1_three_class": {
            "pooled": cls_stats(pred_pooled),
            "routing": cls_stats(pred_routing),
            "top3_hit_rate": {"pooled": top3_hit("pooled"), "routing": top3_hit("routing")},
        },
        "task2_binary_A_vs_BC": {
            "pooled": bin_stats(bin_pred_pooled),
            "routing": bin_stats(bin_pred_routing),
            "auc_final_score": {"pooled": auc_of("top1_final", "pooled"), "routing": auc_of("top1_final", "routing")},
            "auc_semantic_score": {"pooled": auc_of("top1_sem", "pooled"), "routing": auc_of("top1_sem", "routing")},
            "note": "AUC 用 top1 得分计算；final_score 受查询模板权重影响（稳定期规则天花板低），存在天然组间差异",
        },
        "task3_severity_knn": {
            "pooled": knn_reg("pooled"),
            "routing": knn_reg("routing"),
            "note": "以检索 top1 病例的住院天数为严重度预测（与监督实验 Task3 相同的代理标签），LOO 排除自身",
        },
        "group_score_stats_pooled_top1": group_stats,
        "self_in_top_count": sum(1 for r in rows if r.get("self_in_top")),
        "evaluated": len(rows),
    }


# 供 aggregate 使用的病例元数据（住院天数）
CASE_META: dict[str, dict] = {}


if __name__ == "__main__":
    import traceback

    _cases = load_cases()
    for _cid, _c in _cases.items():
        CASE_META[_cid] = {"days": hospital_days(_c["ident"]), "label": _c["label"]}
    main()
