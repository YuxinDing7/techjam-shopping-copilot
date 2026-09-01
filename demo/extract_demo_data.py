import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEMO_DIR = Path(__file__).resolve().parent

SELECTED = {
    "buying": ["public_0106", "public_0027", "public_0031"],
    "browsing": ["public_0195", "public_0173", "public_0121"],
    "intent_override": ["public_0052", "public_0038", "public_0003"],
    "boundary": ["public_0041", "public_0187", "public_0192"],
}
ALL_IDS = {sid for ids in SELECTED.values() for sid in ids}

with open(ROOT / "data/public_set.jsonl") as f:
    public = {}
    for line in f:
        d = json.loads(line)
        if d["sample_id"] in ALL_IDS:
            public[d["sample_id"]] = d

with open(ROOT / "evaluation_results/result_full.json") as f:
    full_meta = {s["sample_id"]: s for s in json.load(f)["sessions"]}
with open(ROOT / "evaluation_results/result_embedding.json") as f:
    emb_meta = {s["sample_id"]: s for s in json.load(f)["sessions"]}


def load_conversation(path):
    sessions = {}
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            if d.get("sample_id") not in ALL_IDS:
                continue
            if "turn" not in d:
                continue
            sessions.setdefault(d["sample_id"], []).append(d)
    for sid in sessions:
        sessions[sid].sort(key=lambda t: t["turn"])
    return sessions


full_conv = load_conversation(ROOT / "evaluation_results/conversation_full.jsonl")
emb_conv = load_conversation(ROOT / "evaluation_results/conversation_embedding.jsonl")

# collect asins we need to resolve
needed_asins = set()
for sid in ALL_IDS:
    needed_asins.add(public[sid]["ground_truth"]["parent_asin"])
    for conv in (full_conv.get(sid, []), emb_conv.get(sid, [])):
        for turn in conv:
            for rec in turn["normalized_recommendations"][:10]:
                needed_asins.add(rec)

catalog = {}
with open(ROOT / "data/catalog.jsonl") as f:
    for line in f:
        d = json.loads(line)
        asin = d["parent_asin"]
        if asin in needed_asins:
            catalog[asin] = {
                "title": d.get("title"),
                "price": d.get("price"),
                "rating": d.get("average_rating"),
                "rating_count": d.get("rating_number"),
                "categories": d.get("categories"),
            }

print("resolved catalog entries:", len(catalog), "of", len(needed_asins))
missing = needed_asins - set(catalog.keys())
if missing:
    print("MISSING:", missing)


def build_turns(conv):
    out = []
    for t in conv:
        recs = t["normalized_recommendations"]
        target = t["target_visible_to_evaluator"]
        rank = recs.index(target) + 1 if target in recs else None
        dbg = t.get("agent_debug", {})
        out.append({
            "turn": t["turn"],
            "user_message": t["user_message"],
            "message": t["agent_response"]["message"],
            "ask_attribute": t["agent_response"].get("ask_attribute"),
            "recommendations": recs[:10],
            "target_rank": rank,
            "hit": t["hit"],
            "route": dbg.get("route"),
            "constraints": [
                f"{c['attribute']}: {c['value']}"
                for c in (dbg.get("active_constraints") or [])
            ],
        })
    return out


demo = {"scenarios": {}}
for scenario, ids in SELECTED.items():
    demo["scenarios"][scenario] = []
    for sid in ids:
        pub = public[sid]
        entry = {
            "sample_id": sid,
            "scenario_type": scenario,
            "difficulty": pub.get("difficulty_bucket"),
            "user_profile": pub["user_profile"],
            "ground_truth": pub["ground_truth"]["parent_asin"],
            "variants": {
                "llm": {
                    "meta": full_meta.get(sid),
                    "turns": build_turns(full_conv.get(sid, [])),
                },
                "embedding": {
                    "meta": emb_meta.get(sid),
                    "turns": build_turns(emb_conv.get(sid, [])),
                },
            },
        }
        demo["scenarios"][scenario].append(entry)

demo["catalog"] = catalog

with open(ROOT / "evaluation_results/result_full.json") as f:
    full_top = json.load(f)
with open(ROOT / "evaluation_results/result_embedding.json") as f:
    emb_top = json.load(f)

def token_breakdown(path):
    ce_prompt = ce_completion = 0
    rerank_prompt = rerank_completion = 0
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            if "turn" not in d:
                continue
            dbg = d.get("agent_debug", {})
            u = (dbg.get("llm_constraint_extraction", {}) or {}).get("usage") or {}
            ce_prompt += u.get("prompt_tokens", 0)
            ce_completion += u.get("completion_tokens", 0)
            for key in ("llm_rerank", "embedding_rerank"):
                block = dbg.get(key)
                if block:
                    u2 = block.get("usage") or {}
                    rerank_prompt += u2.get("prompt_tokens", 0)
                    rerank_completion += u2.get("completion_tokens", 0)
    return {
        "constraint_extraction": {"prompt_tokens": ce_prompt, "completion_tokens": ce_completion},
        "rerank_step": {"prompt_tokens": rerank_prompt, "completion_tokens": rerank_completion},
    }


demo["aggregate"] = {
    "llm": {k: full_top[k] for k in ["sample_count", "hit_rate_at_10", "mrr", "mttc", "efficiency", "recommended_technical_score", "reported_token_usage", "scenario_metrics"]},
    "embedding": {k: emb_top[k] for k in ["sample_count", "hit_rate_at_10", "mrr", "mttc", "efficiency", "recommended_technical_score", "reported_token_usage", "scenario_metrics"]},
}
demo["aggregate"]["llm"]["token_breakdown"] = token_breakdown(ROOT / "evaluation_results/conversation_full.jsonl")
demo["aggregate"]["embedding"]["token_breakdown"] = token_breakdown(ROOT / "evaluation_results/conversation_embedding.jsonl")

data_path = DEMO_DIR / "demo_data.json"
with open(data_path, "w") as f:
    json.dump(demo, f, ensure_ascii=False)

# Merge with the template to produce the final standalone page.
data_json = json.dumps(demo, ensure_ascii=False).replace("</", "<\\/")
template = (DEMO_DIR / "demo_template.html").read_text()
html_path = DEMO_DIR / "demo.html"
html_path.write_text(template.replace("__DEMO_DATA_JSON__", data_json))

import os
print("wrote", data_path, f"({os.path.getsize(data_path) / 1024:.1f} KB)")
print("wrote", html_path, f"({os.path.getsize(html_path) / 1024:.1f} KB)")
