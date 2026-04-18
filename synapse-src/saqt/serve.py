#!/usr/bin/env python3
"""
SAQT Server — Minimal
======================
Sentence transformer + Q&A pairs + sandbox. Nothing else.
All intelligence lives in the knowledge base.
"""

import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from http.server import HTTPServer, SimpleHTTPRequestHandler
import json, os, time, re, subprocess
from pathlib import Path

DEVICE = "cpu"
HOME = os.environ.get("HOME", "/home/tejasphatak")
PORT = int(os.environ.get("SAQT_PORT", "3000"))
PUBLIC_DIR = Path(__file__).parent / "public"
QA_PATH = os.environ.get("SAQT_QA",
    os.path.join(HOME, "webmind-research/trained_model/qa_pairs.jsonl"))
EMB_PATH = os.environ.get("SAQT_EMB",
    os.path.join(HOME, "webmind-research/trained_model/qa_embeddings.pt"))


class SAQTEngine:
    def __init__(self):
        print("[saqt] Loading encoder...", flush=True)
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2', device=DEVICE)

        print("[saqt] Loading knowledge base...", flush=True)
        self.pairs = []
        with open(QA_PATH) as f:
            for line in f:
                self.pairs.append(json.loads(line))

        self.embeddings = torch.load(EMB_PATH, map_location=DEVICE, weights_only=True)
        print(f"[saqt] {len(self.pairs):,} pairs, {self.embeddings.shape}", flush=True)

    def query(self, question, max_hops=5):
        t0 = time.time()
        q_emb = self.encoder.encode([question], convert_to_tensor=True,
                                   show_progress_bar=False)[0]

        visited = set()
        facts, answers, trace = [], [], []

        for hop in range(max_hops):
            # Search
            sims = F.cosine_similarity(q_emb.unsqueeze(0), self.embeddings)
            top_vals, top_idxs = sims.topk(min(3 + len(visited), len(self.pairs)))

            for val, idx in zip(top_vals, top_idxs):
                i = idx.item()
                if i in visited:
                    continue
                if val.item() < 0.2:
                    break
                visited.add(i)
                pair = self.pairs[i]
                a = pair.get("answer", "")
                q = pair.get("question", "")
                if a and a not in answers:
                    answers.append(a)
                if q and q not in facts:
                    facts.append(q)
                break  # One new fact per hop

            trace.append({"hop": hop, "score": round(top_vals[0].item(), 3),
                         "facts": len(facts)})

            # Re-encode with context
            ctx = f"{question} {' '.join(answers[-3:])}"
            q_emb = self.encoder.encode([ctx], convert_to_tensor=True,
                                       show_progress_bar=False)[0]

            # Convergence: if top score barely changed, stop
            if hop > 0 and abs(trace[-1]["score"] - trace[-2]["score"]) < 0.01:
                break

        # Best answer
        answer = answers[0] if answers else ""

        # Tool call: <tool> tag → substitute {QUERY} → sandbox
        if '<tool>' in answer:
            match = re.search(r'<tool>(.*?)</tool>', answer, re.DOTALL)
            if match:
                code = match.group(1).strip().replace('{QUERY}', question)
                try:
                    result = subprocess.run(
                        ["python3", "-c", code],
                        capture_output=True, text=True, timeout=5,
                        env={"PATH": "/usr/bin:/bin", "HOME": "/tmp"},
                        cwd="/tmp")
                    if result.stdout.strip():
                        answer = result.stdout.strip()
                except:
                    pass

        return {
            "question": question, "answer": answer,
            "facts": facts[:10], "answers": answers[:5], "trace": trace,
            "hops": len(trace), "totalFacts": len(facts),
            "timeMs": int((time.time() - t0) * 1000),
        }

    def stats(self):
        return {"ready": True, "chunks": len(self.pairs), "device": DEVICE}


# ── HTTP ──────────────────────────────────────────────────

engine = None

class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *a, **kw):
        super().__init__(*a, directory=str(PUBLIC_DIR), **kw)

    def do_OPTIONS(self):
        self.send_response(204)
        self._cors()
        self.end_headers()

    def do_GET(self):
        if self.path == '/api/saqt/stats':
            self._json(engine.stats())
        elif self.path in ('/', '/chat.html'):
            self.path = '/chat.html'
            super().do_GET()
        else:
            super().do_GET()

    def do_POST(self):
        if self.path == '/api/saqt/query':
            body = json.loads(self.rfile.read(int(self.headers.get('Content-Length', 0))))
            q = body.get('question', '')
            ctx = body.get('context', '')
            if not q:
                self._json({"error": "Missing question"}, 400)
                return
            full = f"{ctx} {q}".strip() if ctx else q
            result = engine.query(full, max_hops=min(body.get('hops', 5), 10))
            result['question'] = q
            self._json(result)
        else:
            self.send_error(404)

    def _json(self, data, code=200):
        self.send_response(code)
        self._cors()
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())

    def _cors(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET,POST,OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')

    def log_message(self, fmt, *a):
        if '/api/' in str(a[0]):
            print(f"[saqt] {a[0]}", flush=True)


if __name__ == "__main__":
    engine = SAQTEngine()
    print(f"[saqt] http://0.0.0.0:{PORT}", flush=True)
    HTTPServer(('0.0.0.0', PORT), Handler).serve_forever()
