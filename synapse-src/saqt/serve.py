#!/usr/bin/env python3
"""
SAQT Server — Real distributed cognition demo
===============================================
Python backend with actual sentence transformer + trained kernel.
Serves chat.html frontend via HTTP.

This IS the SAQT system from the paper:
- Sentence transformer for embedding + routing
- Trained reasoning kernel at each hop
- Multi-hop traversal across distributed neurons
- Facts + reasoning trace returned to frontend
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from transformers import GPT2Config, GPT2LMHeadModel, GPT2Tokenizer
from http.server import HTTPServer, SimpleHTTPRequestHandler
import json, os, time, threading
from pathlib import Path

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
HOME = os.environ.get("HOME", "/home/tejasphatak")
PORT = int(os.environ.get("SAQT_PORT", "3000"))
PUBLIC_DIR = Path(__file__).parent / "public"
CHUNKS_PATH = os.environ.get("SAQT_CHUNKS",
    os.path.join(HOME, "webmind-research/trained_model/chunks.jsonl"))
KERNEL_PATH = os.environ.get("SAQT_KERNEL",
    os.path.join(HOME, "webmind-research/trained_model/kernel.pt"))
N_NEURONS = 20
REPLICATION = 3


# ══════════════════════════════════════════════════════════════
# SAQT Engine
# ══════════════════════════════════════════════════════════════

class SAQTEngine:
    def __init__(self):
        print("[saqt] Loading sentence transformer...", flush=True)
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2', device=DEVICE)

        print("[saqt] Loading tokenizer...", flush=True)
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load kernel
        self.kernel = None
        if os.path.exists(KERNEL_PATH):
            print(f"[saqt] Loading trained kernel from {KERNEL_PATH}...", flush=True)
            config = GPT2Config(vocab_size=50257, n_positions=256,
                               n_embd=256, n_layer=2, n_head=4, n_inner=1024)
            self.kernel = GPT2LMHeadModel(config).to(DEVICE)
            state = torch.load(KERNEL_PATH, map_location=DEVICE, weights_only=True)
            if any(k.startswith("model.") for k in state):
                state = {k.replace("model.", "", 1): v for k, v in state.items()}
            self.kernel.load_state_dict(state)
            self.kernel.eval()
            print("[saqt] Kernel loaded", flush=True)

        # Load chunks
        print(f"[saqt] Loading chunks from {CHUNKS_PATH}...", flush=True)
        self.chunks = []
        with open(CHUNKS_PATH) as f:
            for line in f:
                self.chunks.append(json.loads(line))
        print(f"[saqt] {len(self.chunks):,} chunks loaded", flush=True)

        # Load or encode embeddings
        emb_path = os.path.join(HOME, "webmind-research/trained_model/embeddings.pt")
        if os.path.exists(emb_path):
            print(f"[saqt] Loading pre-encoded embeddings from {emb_path}...", flush=True)
            self.embeddings = torch.load(emb_path, map_location=DEVICE, weights_only=True)
            # Check size match
            if self.embeddings.size(0) != len(self.chunks):
                print(f"[saqt] Embedding count mismatch ({self.embeddings.size(0)} vs {len(self.chunks)}), re-encoding...", flush=True)
                self._encode_chunks()
            else:
                print(f"[saqt] Loaded {self.embeddings.size(0)} embeddings", flush=True)
        else:
            self._encode_chunks()

        # Distribute to neurons
        self._build_neurons()
        print(f"[saqt] Ready. {len(self.chunks):,} chunks, {N_NEURONS} neurons", flush=True)

    def _encode_chunks(self):
        print("[saqt] Encoding chunks (this takes a few minutes on CPU)...", flush=True)
        t0 = time.time()
        texts = [c["text"] for c in self.chunks]
        self.embeddings = self.encoder.encode(texts, convert_to_tensor=True,
                                              batch_size=256, show_progress_bar=True).to(DEVICE)
        print(f"[saqt] Encoded in {time.time()-t0:.0f}s", flush=True)

    def _build_neurons(self):
        import random
        topic_groups = {}
        for i, c in enumerate(self.chunks):
            t = c.get("topic", c.get("category", "general"))
            topic_groups.setdefault(t, []).append(i)

        self.neuron_data = {i: [] for i in range(N_NEURONS)}
        for j, (topic, indices) in enumerate(sorted(topic_groups.items())):
            primary = j % N_NEURONS
            self.neuron_data[primary].extend(indices)
            replicas = random.sample(
                [n for n in range(N_NEURONS) if n != primary],
                min(REPLICATION - 1, N_NEURONS - 1))
            for r in replicas:
                self.neuron_data[r].extend(indices)

        self.profiles = {}
        for nid, indices in self.neuron_data.items():
            if indices:
                self.profiles[nid] = self.embeddings[indices].mean(dim=0)

    def query(self, question, max_hops=5):
        t0 = time.time()
        q_emb = self.encoder.encode([question], convert_to_tensor=True,
                                   show_progress_bar=False)[0].to(DEVICE)

        facts = []
        answers = []
        trace = []
        path = []

        for hop in range(max_hops):
            # Route to best unvisited neuron
            best_nid, best_sim = None, -1
            for nid, profile in self.profiles.items():
                if nid in path:
                    continue
                sim = F.cosine_similarity(
                    q_emb.unsqueeze(0), profile.unsqueeze(0)).item()
                if sim > best_sim:
                    best_sim = sim
                    best_nid = nid

            if best_nid is None:
                break

            # Retrieve from this neuron
            n_indices = self.neuron_data[best_nid]
            n_embs = self.embeddings[n_indices]
            sims = F.cosine_similarity(q_emb.unsqueeze(0), n_embs)
            top_k = min(3, len(n_indices))
            top_vals, top_idxs = sims.topk(top_k)

            hop_facts = []
            for j, idx in enumerate(top_idxs):
                if top_vals[j] > 0.2:
                    chunk = self.chunks[n_indices[idx.item()]]
                    text = chunk["text"]
                    if text not in facts:
                        facts.append(text)
                        hop_facts.append(text[:100])
                    if chunk.get("answer") and chunk["answer"] not in answers:
                        answers.append(chunk["answer"])

            # Reasoning kernel (optional — skip if it crashes on CPU)
            thought = ""
            if self.kernel:
                try:
                    ctx = f"Question: {question} Facts: {' | '.join(facts[-4:])}"
                    inp = self.tokenizer(ctx, return_tensors='pt', truncation=True,
                                        max_length=200).to(DEVICE)
                    with torch.no_grad():
                        out = self.kernel.generate(inp['input_ids'],
                                                 max_new_tokens=20, pad_token_id=50256)
                    new_tokens = out[0][inp['input_ids'].size(1):]
                    thought = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
                except Exception as e:
                    thought = ""

            trace.append({
                "hop": hop,
                "neuron": best_nid,
                "similarity": round(best_sim, 3),
                "factsFound": len(hop_facts),
                "thought": thought[:80] if thought else ""
            })

            path.append(best_nid)

            # Re-encode with accumulated context
            fact_str = " | ".join(facts[-6:])
            trace_str = " -> ".join([t["thought"] for t in trace if t["thought"]])
            full_ctx = f"Question: {question} Facts: {fact_str}"
            if trace_str:
                full_ctx += f" Reasoning: {trace_str}"
            q_emb = self.encoder.encode([full_ctx], convert_to_tensor=True,
                                       show_progress_bar=False)[0].to(DEVICE)

        elapsed_ms = int((time.time() - t0) * 1000)

        # Synthesize answer from retrieved facts
        answer = ""
        if answers:
            answer = answers[0]
        elif facts:
            answer = facts[0]

        # Try to synthesize a coherent answer using retrieved facts
        answer = self._synthesize(question, facts, answers)

        return {
            "question": question,
            "answer": answer,
            "facts": facts[:10],
            "answers": answers[:5],
            "trace": trace,
            "hops": len(path),
            "path": path,
            "totalFacts": len(facts),
            "timeMs": elapsed_ms,
        }

    def _synthesize(self, question, facts, answers):
        """Synthesize a coherent answer from retrieved facts.
        Uses Gemini Flash as temporary bridge until kernel is trained.
        TODO: Replace with trained GPT-2 125M kernel."""
        if not facts and not answers:
            return "I don't have enough information in my knowledge base to answer that."

        # Try Gemini Flash for synthesis (temporary — will be replaced by local kernel)
        gemini_key = ""
        try:
            gpath = os.path.join(HOME, ".claude/secrets/gemini.json")
            if os.path.exists(gpath):
                gemini_key = json.loads(open(gpath).read()).get("api_key", "")
        except:
            pass

        if gemini_key:
            try:
                import google.generativeai as genai
                genai.configure(api_key=gemini_key)
                model = genai.GenerativeModel('gemini-2.5-flash')
                context = "\n".join(facts[:6])
                direct = f"\nDirect answers: {', '.join(answers[:3])}" if answers else ""
                prompt = f"Answer the question using ONLY the provided facts. Be concise, natural, and helpful. If the facts don't contain the answer, say so honestly. Do not add information beyond what the facts state.\n\nFacts:\n{context}{direct}\n\nQuestion: {question}\n\nAnswer:"
                resp = model.generate_content(prompt,
                    generation_config={'max_output_tokens': 300, 'temperature': 0.3})
                if resp.text:
                    return resp.text
            except Exception as e:
                print(f"[saqt] Synthesis fallback: {e}", flush=True)

        # Fallback: return best answer or fact directly
        if answers:
            return answers[0]
        return facts[0] if facts else "No answer found."

    def stats(self):
        return {
            "ready": True,
            "chunks": len(self.chunks),
            "neurons": N_NEURONS,
            "replication": REPLICATION,
            "device": DEVICE,
            "hasKernel": self.kernel is not None,
            "sources": list(set(c.get("source", "") for c in self.chunks)),
            "topics": len(set(c.get("topic", "") for c in self.chunks)),
        }


# ══════════════════════════════════════════════════════════════
# HTTP Server
# ══════════════════════════════════════════════════════════════

engine = None

class SAQTHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(PUBLIC_DIR), **kwargs)

    def do_OPTIONS(self):
        self.send_response(204)
        self._cors()
        self.end_headers()

    def do_GET(self):
        if self.path == '/api/saqt/stats':
            self._json_response(engine.stats())
        elif self.path == '/' or self.path == '/chat.html':
            self.path = '/chat.html'
            super().do_GET()
        else:
            super().do_GET()

    def do_POST(self):
        if self.path == '/api/saqt/query':
            length = int(self.headers.get('Content-Length', 0))
            body = json.loads(self.rfile.read(length))
            question = body.get('question', '')
            hops = min(body.get('hops', 5), 10)
            if not question:
                self._json_response({"error": "Missing question"}, 400)
                return
            result = engine.query(question, max_hops=hops)
            self._json_response(result)
        else:
            self.send_error(404)

    def _json_response(self, data, code=200):
        self.send_response(code)
        self._cors()
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())

    def _cors(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET,POST,OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')

    def log_message(self, format, *args):
        if '/api/' in str(args[0]):
            print(f"[saqt] {args[0]}", flush=True)


if __name__ == "__main__":
    print("=== SAQT Server ===\n", flush=True)
    engine = SAQTEngine()
    print(f"\n[saqt] Starting server on http://0.0.0.0:{PORT}", flush=True)
    server = HTTPServer(('0.0.0.0', PORT), SAQTHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[saqt] Stopped", flush=True)
