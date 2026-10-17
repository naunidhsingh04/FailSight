import os, sys, json, csv, re, statistics as st
root = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.abspath(__file__))
os.makedirs("stats_out", exist_ok=True)
STATUS = ["status_code","status","http_status","statusCode"]
LAT = ["latency","latency_ms","response_time","elapsed","duration","elapsed_ms"]
TS = ["timestamp","time","ts","start_time"]
pick = lambda r, ks: next((r[k] for k in ks if k in r), None)
rows, api = [], []
for sysn in ("SN_data", "TT_data"):
    for dt in sorted(os.listdir(os.path.join(root, sysn))):
        p = os.path.join(root, sysn, dt)
        if not os.path.isdir(p): continue
        for sc in sorted(os.listdir(p)):
            sp = os.path.join(p, sc)
            if not os.path.isdir(sp): continue
            files = [os.path.join(d, f) for d, _, fs in os.walk(sp) for f in fs if f != ".DS_Store"]
            size = sum(os.path.getsize(f) for f in files)
            lines = 0
            for f in files:
                if f.lower().endswith((".jsonl", ".log", ".txt", ".csv")):
                    with open(f, "rb") as fh: lines += sum(1 for _ in fh)
            rows.append([sysn, dt, sc, len(files), round(size/1e6, 2), lines])
            if dt == "api_responses":
                for f in files:
                    if f.endswith(".jsonl"):
                        recs = [json.loads(l) for l in open(f, encoding="utf-8") if l.strip()]
                        if not recs: continue
                        if len(api) == 0: print("API KEYS:", list(recs[0].keys()))
                        sc_ = [pick(r, STATUS) for r in recs]
                        lt = [float(x) for x in (pick(r, LAT) for r in recs) if isinstance(x, (int, float))]
                        ts = [x for x in (pick(r, TS) for r in recs) if x is not None]
                        api.append([sysn, sc, len(recs),
                            sum(1 for s in sc_ if s and 200 <= int(s) < 300),
                            sum(1 for s in sc_ if s and 400 <= int(s) < 500),
                            sum(1 for s in sc_ if s and int(s) >= 500),
                            round(st.mean(lt), 2) if lt else "",
                            round(sorted(lt)[int(.95*len(lt))-1], 2) if lt else "",
                            min(ts) if ts else "", max(ts) if ts else ""])
csv.writer(open("stats_out/inventory.csv", "w", newline="")).writerows(
    [["system","datatype","scenario","files","size_MB","lines"]] + rows)
csv.writer(open("stats_out/api_stats.csv", "w", newline="")).writerows(
    [["system","scenario","requests","2xx","4xx","5xx","lat_mean","lat_p95","first_ts","last_ts"]] + api)
print("done: stats_out/inventory.csv and stats_out/api_stats.csv")