#!/usr/bin/env python3
"""
SAQT Server — Minimal. FAISS + weights + confidence.
Search + <tool> sandbox + feedback.
All intelligence in the knowledge base.
"""

import numpy as np
import faiss
import sqlite3
import json, os, time, re, subprocess
from pathlib import Path
from http.server import HTTPServer, SimpleHTTPRequestHandler
from sentence_transformers import SentenceTransformer

HOME = os.environ.get("HOME", "/home/tejasphatak")
PORT = int(os.environ.get("SAQT_PORT", "3000"))
PUBLIC_DIR = Path(__file__).parent / "public"
DB_PATH = os.environ.get("SAQT_DB",
    os.path.join(HOME, "webmind-research/trained_model/saqt.db"))
INDEX_PATH = os.environ.get("SAQT_INDEX",
    os.path.join(HOME, "webmind-research/trained_model/saqt.faiss"))
# Fallback to JSONL if no DB
QA_PATH = os.path.join(HOME, "webmind-research/trained_model/qa_pairs.jsonl")
EMB_PATH = os.path.join(HOME, "webmind-research/trained_model/qa_embeddings.pt")
CONFIDENCE_THRESHOLD = 0.35


class SAQTEngine:
    def __init__(self):
        print("[saqt] Loading encoder...", flush=True)
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')

        # Try FAISS+SQLite first, fall back to JSONL+torch
        if os.path.exists(DB_PATH) and os.path.exists(INDEX_PATH):
            print(f"[saqt] Loading FAISS + SQLite...", flush=True)
            self.index = faiss.read_index(INDEX_PATH)
            self.db = sqlite3.connect(DB_PATH, check_same_thread=False)
            self.mode = "faiss"
            count = self.db.execute("SELECT COUNT(*) FROM qa").fetchone()[0]
            print(f"[saqt] {count:,} pairs in FAISS+SQLite", flush=True)
        else:
            print(f"[saqt] Loading JSONL+torch fallback...", flush=True)
            import torch, torch.nn.functional as F
            self.torch = torch
            self.F = F
            self.pairs = []
            with open(QA_PATH) as f:
                for line in f: self.pairs.append(json.loads(line))
            self.embeddings = torch.load(EMB_PATH, map_location='cpu', weights_only=True)
            self.mode = "torch"
            print(f"[saqt] {len(self.pairs):,} pairs in torch mode", flush=True)

    def search(self, query, top_k=5):
        q_emb = self.encoder.encode([query], normalize_embeddings=True).astype(np.float32)

        if self.mode == "faiss":
            scores, indices = self.index.search(q_emb, top_k * 3)
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < 0: continue
                row = self.db.execute(
                    "SELECT id, question, answer, source, weight FROM qa WHERE id=?",
                    (int(idx) + 1,)).fetchone()
                if row:
                    results.append({
                        "id": row[0], "question": row[1], "answer": row[2],
                        "source": row[3], "weight": row[4] or 1.0,
                        "raw_score": float(score),
                        "score": float(score) * (row[4] or 1.0),
                    })
            results.sort(key=lambda x: x["score"], reverse=True)
            return results[:top_k]
        else:
            q_t = self.torch.tensor(q_emb)
            sims = self.F.cosine_similarity(q_t, self.embeddings)
            top_vals, top_idxs = sims.topk(top_k)
            return [{"id": idx.item(), "question": self.pairs[idx.item()].get("question",""),
                     "answer": self.pairs[idx.item()].get("answer",""),
                     "source": self.pairs[idx.item()].get("source",""),
                     "score": val.item(), "weight": 1.0}
                    for val, idx in zip(top_vals, top_idxs)]

    def boost(self, pair_id):
        if self.mode == "faiss":
            self.db.execute("UPDATE qa SET weight = MIN(weight * 1.1, 5.0) WHERE id=?", (pair_id,))
            self.db.commit()

    def penalize(self, pair_id):
        if self.mode == "faiss":
            self.db.execute("UPDATE qa SET weight = MAX(weight * 0.9, 0.1) WHERE id=?", (pair_id,))
            self.db.commit()

    def extract_topic(self, query):
        """Two-pass query understanding: extract topic from format/action requests."""
        patterns = [
            # "create/make/write a markdown/list about/for TOPIC"
            r'(?:can you |please |could you |i need |i want )?(?:create|make|write|generate|draft|prepare|give me|provide|build|compose|put together|show me)(?:\s+me)?\s+(?:a|an|the|some)?\s*(?:markdown|md|list|table|document|doc|summary|report|essay|article|outline|presentation|slides?|spreadsheet|csv|json|html|text|paragraph|bullets?|overview|brief|writeup|write-up|notes?|chart|graph|diagram)\s*(?:about|for|of|on|regarding|related to|covering|explaining|describing|summarizing|detailing)\s+(.+)',
            # "TOPIC in markdown format"
            r'(.+?)\s+(?:in|using|as|formatted as|formatted in)\s+(?:a\s+)?(?:markdown|md|list|table|document|summary|report|essay|article|outline|bullets?|html|text|paragraph)\s*(?:format)?$',
            # "summarize/explain TOPIC"
            r'(?:can you |please |could you )?(?:summarize|explain|describe|elaborate on|tell me about|give me info on|give me information about|what do you know about)\s+(.+)',
        ]
        for pattern in patterns:
            match = re.match(pattern, query, re.IGNORECASE)
            if match and match.group(1):
                topic = match.group(1).rstrip('?.!,').strip()
                if len(topic) > 2:
                    return topic
        return None

    def query(self, question, max_hops=5):
        t0 = time.time()

        # Two-pass: extract topic if format/action request
        topic = self.extract_topic(question)
        search_query = question

        if topic:
            topic_results = self.search(topic, top_k=1)
            full_results = self.search(question, top_k=1)
            topic_score = topic_results[0]["score"] if topic_results else 0
            full_score = full_results[0]["score"] if full_results else 0

            AMBIGUITY_THRESHOLD = 0.5
            topic_strong = topic_score >= AMBIGUITY_THRESHOLD
            full_strong = full_score >= AMBIGUITY_THRESHOLD

            # Check if matches are about different topics
            match_similarity = 1.0
            if topic_results[0]["id"] != full_results[0]["id"]:
                t_emb = self.encoder.encode([topic_results[0]["question"]], normalize_embeddings=True)
                f_emb = self.encoder.encode([full_results[0]["question"]], normalize_embeddings=True)
                match_similarity = float(np.dot(t_emb[0], f_emb[0]))

            # Both strong + matches about different topics (low similarity) → ambiguous
            if (topic_strong and full_strong and match_similarity < 0.5):
                return {
                    "question": question,
                    "answer": (
                        f"I found strong matches for different interpretations of your question:\n\n"
                        f"1. **{topic}** — \"{topic_results[0]['question'][:80]}\"\n"
                        f"2. **{question}** — \"{full_results[0]['question'][:80]}\"\n\n"
                        f"Could you clarify what you're looking for?"
                    ),
                    "confidence": max(topic_score, full_score),
                    "ambiguous": True,
                    "facts": [], "answers": [], "trace": [],
                    "hops": 0, "totalFacts": 0,
                    "timeMs": int((time.time() - t0) * 1000),
                }

            # One strong → use it
            if topic_strong:
                search_query = topic

        results = self.search(search_query, top_k=3)

        if not results or results[0]["score"] < CONFIDENCE_THRESHOLD:
            return {
                "question": question,
                "answer": "I don't have enough confidence to answer that. My best match was too weak.",
                "confidence": results[0]["score"] if results else 0,
                "facts": [], "answers": [], "trace": [],
                "hops": 0, "totalFacts": 0,
                "timeMs": int((time.time() - t0) * 1000),
            }

        best = results[0]
        answer = best["answer"]
        facts = [r["question"] for r in results if r["question"]]
        answers = [r["answer"] for r in results if r["answer"]]

        # Multi-hop: re-encode with context, search again
        trace = [{"hop": 0, "score": best["score"], "id": best["id"]}]
        visited = {best["id"]}

        for hop in range(1, max_hops):
            ctx = f"{question} {answer[:200]}"
            new_results = self.search(ctx, top_k=5)
            found_new = False
            for r in new_results:
                if r["id"] not in visited and r["score"] > CONFIDENCE_THRESHOLD:
                    visited.add(r["id"])
                    if r["answer"] not in answers:
                        answers.append(r["answer"])
                    if r["question"] not in facts:
                        facts.append(r["question"])
                    trace.append({"hop": hop, "score": r["score"], "id": r["id"]})
                    found_new = True
                    break
            if not found_new:
                break

        # Tool call
        if '<tool>' in answer:
            match = re.search(r'<tool>(.*?)</tool>', answer, re.DOTALL)
            if match:
                code = match.group(1).strip().replace('{QUERY}', question)
                try:
                    result = subprocess.run(
                        ["python3", "-c", code],
                        capture_output=True, text=True, timeout=5,
                        env={"PATH": "/usr/bin:/bin", "HOME": "/tmp"}, cwd="/tmp")
                    if result.stdout.strip():
                        answer = result.stdout.strip()
                except:
                    pass

        return {
            "question": question, "answer": answer,
            "confidence": best["score"],
            "facts": facts[:10], "answers": answers[:5], "trace": trace,
            "hops": len(trace), "totalFacts": len(facts),
            "timeMs": int((time.time() - t0) * 1000),
            "matchId": best["id"],
        }

    def stats(self):
        if self.mode == "faiss":
            count = self.db.execute("SELECT COUNT(*) FROM qa").fetchone()[0]
        else:
            count = len(self.pairs)
        return {"ready": True, "chunks": count, "mode": self.mode}


# ── HTTP ──────────────────────────────────────────────────

engine = None

class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *a, **kw):
        super().__init__(*a, directory=str(PUBLIC_DIR), **kw)

    def do_OPTIONS(self):
        self.send_response(204); self._cors(); self.end_headers()

    def do_GET(self):
        if self.path == '/api/saqt/stats':
            self._json(engine.stats())
        elif self.path in ('/', '/chat.html'):
            self.path = '/chat.html'; super().do_GET()
        else:
            super().do_GET()

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers.get('Content-Length', 0))))
        # OpenAI-compatible chat completions endpoint
        if self.path == '/v1/chat/completions':
            messages = body.get('messages', [])
            # Extract the last user message
            q = ''
            for m in reversed(messages):
                if m.get('role') == 'user':
                    content = m.get('content', '')
                    if isinstance(content, list):
                        content = ' '.join(c.get('text', '') for c in content if c.get('type') == 'text')
                    q = content
                    break
            if not q:
                self._json({"error": {"message": "No user message"}}, 400)
                return
            result = engine.query(q, max_hops=5)
            # Format as OpenAI response
            self._json({
                "id": f"wmind-{int(time.time())}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": "webmind-305k",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": result["answer"]},
                    "finish_reason": "stop"
                }],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                "_webmind": {
                    "confidence": result.get("confidence", 0),
                    "hops": result.get("hops", 0),
                    "timeMs": result.get("timeMs", 0),
                    "matchId": result.get("matchId"),
                }
            })
            return
        elif self.path == '/api/saqt/query':
            q = body.get('question', '')
            ctx = body.get('context', '')
            if not q: self._json({"error": "Missing question"}, 400); return
            full = f"{ctx} {q}".strip() if ctx else q
            result = engine.query(full, max_hops=min(body.get('hops', 5), 10))
            result['question'] = q
            self._json(result)
        elif self.path == '/api/saqt/feedback':
            pair_id = body.get('id')
            action = body.get('action')  # "boost" or "penalize"
            if pair_id and action == 'boost':
                engine.boost(pair_id)
                self._json({"ok": True})
            elif pair_id and action == 'penalize':
                engine.penalize(pair_id)
                self._json({"ok": True})
            else:
                self._json({"error": "Need id + action"}, 400)
        else:
            self.send_error(404)

    def _json(self, data, code=200):
        self.send_response(code); self._cors()
        self.send_header('Content-Type', 'application/json'); self.end_headers()
        self.wfile.write(json.dumps(data).encode())

    def _cors(self):
        for h, v in [('Access-Control-Allow-Origin', '*'),
                      ('Access-Control-Allow-Methods', 'GET,POST,OPTIONS'),
                      ('Access-Control-Allow-Headers', 'Content-Type, Authorization')]:
            self.send_header(h, v)

    def log_message(self, fmt, *a):
        if '/api/' in str(a[0]): print(f"[saqt] {a[0]}", flush=True)


if __name__ == "__main__":
    engine = SAQTEngine()
    print(f"[saqt] http://0.0.0.0:{PORT}", flush=True)
    HTTPServer(('0.0.0.0', PORT), Handler).serve_forever()
