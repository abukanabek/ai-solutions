import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment as hungarian
from tqdm.auto import tqdm
from itertools import chain
import random
# ------------------------------------------------------------------------------
train = pd.read_csv("train_data.csv")
test = pd.read_csv("test_data.csv")
subm = pd.read_csv("sample_output.csv")

train.shape, test.shape, subm.shape
# ------------------------------------------------------------------------------
train.head()
# ------------------------------------------------------------------------------
test.head()
# ------------------------------------------------------------------------------
import csv

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

MODEL_NAME = "google-bert/bert-base-uncased"
EMBED_DIM = 768
HIDDEN_DIM = 256
PROJ_DIM = 512
BATCH_SIZE = 25
DEVICE = torch.device('cuda' if torch.cuda.is_available() else "cpu")

MAX_LENGTH = 256
# ------------------------------------------------------------------------------
def load_tokenizer_and_model(MODEL_NAME):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE)
    return tokenizer, model

def encode_batch(model, tokenizer, texts, device, batch_size=BATCH_SIZE):
    all_embs = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Encoding"):
        batch = texts[i: i + batch_size]
        enc = tokenizer(
            batch,
            max_length=MAX_LENGTH,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        with torch.no_grad():
            out = model(
                input_ids=enc["input_ids"].to(device),
                attention_mask=enc["attention_mask"].to(device),
            )
        last_hidden = out.last_hidden_state
        mask = enc["attention_mask"].unsqueeze(-1).float().to(device)
        embs = (last_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        all_embs.append(embs.cpu())
    return torch.cat(all_embs, dim=0)

def load_train_csv(path):
    df = pd.read_csv(path)
    python_codes, c_codes = df['py_source'].tolist(), df['cpp_source'].tolist()
    return python_codes, c_codes
# ------------------------------------------------------------------------------
tokenizer, model = load_tokenizer_and_model(MODEL_NAME)
# ------------------------------------------------------------------------------
py_codes, c_codes = load_train_csv('train_data.csv')

py_embs = encode_batch(model, tokenizer, py_codes, DEVICE)
c_embs = encode_batch(model, tokenizer, c_codes, DEVICE)
# ------------------------------------------------------------------------------
class ProjectionHead(nn.Module):
    def __init__(self, embed_dim, proj_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(embed_dim, proj_dim),
            nn.ReLU(),
            nn.Linear(proj_dim, proj_dim)
        )
        
    def forward(self, x):
        return self.layers(x)
# ------------------------------------------------------------------------------
def fit(py_proj, c_proj, py_embs, c_embs, criterion, optimizer, device, epochs=10, log_rate=1, batch_size=BATCH_SIZE, history=None):
    if history is None:
        history = {
            'tloss': [],
            'vloss': []
        }
    
    for epoch in tqdm(range(epochs), desc='Epoch'):
        py_proj.train()
        c_proj.train()
        rloss, i = 0, 0
        for it in range(batch_size, len(py_embs), batch_size):
            py_batch, c_batch = py_embs[it:it+batch_size], c_embs[it:it+batch_size]
            py_batch, c_batch = py_batch.to(device), c_batch.to(device)

            py_output = py_proj(py_batch)
            c_output = c_proj(c_batch)
            target = torch.ones(py_output.shape[0]).to(device)
            loss = criterion(py_output, c_output, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            rloss += loss.item()
            i += 1
            cur_loss = rloss/i

            # pbar.set_postfix({'loss': f'{cur_loss:.5f}'})
            
        for it in range(batch_size, len(py_embs), batch_size):
            c_indices = []
            for itt in range(it, it+batch_size):
                chosen = random.randint(batch_size, len(py_embs)-1)
                while chosen == itt:
                    chosen = random.randint(batch_size, len(py_embs)-1)
                c_indices.append(chosen)
                
            c_indices = torch.tensor(c_indices)
            
            py_batch, c_batch = py_embs[it:it+batch_size], c_embs[c_indices]
            py_batch, c_batch = py_batch.to(device), c_batch.to(device)

            py_output = py_proj(py_batch)
            c_output = c_proj(c_batch)
            target = -torch.ones(py_output.shape[0]).to(device)
            loss = criterion(py_output, c_output, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            rloss += loss.item()
            i += 1
            cur_loss = rloss/i

            # pbar.set_postfix({'loss': f'{cur_loss:.5f}'})
            
        history['tloss'].append(cur_loss)
        if (epoch+1) % log_rate == 0:
            print(f'Epoch: {epoch+1}/{epochs} | Train Loss: {cur_loss:.5f}')

        py_proj.eval()
        c_proj.eval()
        rloss, i = 0, 0
        for it in range(0, batch_size, batch_size):
            py_batch, c_batch = py_embs[it:it+batch_size], c_embs[it:it+batch_size]
            py_batch, c_batch = py_batch.to(device), c_batch.to(device)

            with torch.no_grad():
                py_output = py_proj(py_batch)
                c_output = c_proj(c_batch)
                target = torch.ones(py_output.shape[0]).to(device)
                loss = criterion(py_output, c_output, target)

            rloss += loss.item()
            i += 1
            cur_loss = rloss/i

            # pbar.set_postfix({'loss': f'{cur_loss:.5f}'})
            
        history['vloss'].append(cur_loss)
        if (epoch+1) % log_rate == 0:
            print(f'Valid Loss: {cur_loss:.5f}')
            
    return history
# ------------------------------------------------------------------------------
py_proj = ProjectionHead(EMBED_DIM, PROJ_DIM)
c_proj = ProjectionHead(EMBED_DIM, PROJ_DIM)

optimizer = torch.optim.AdamW(chain(py_proj.parameters(), c_proj.parameters()), lr=1e-5)
criterion = nn.CosineEmbeddingLoss()
# ------------------------------------------------------------------------------
history = fit(py_proj, c_proj, py_embs, c_embs, criterion, optimizer, DEVICE, epochs=4000, log_rate=1000)
# ------------------------------------------------------------------------------
plt.figure(figsize=(10, 3))

plt.plot(history['tloss'], label='train')
plt.plot(history['vloss'], label='valid')

plt.legend()
plt.show()
# ------------------------------------------------------------------------------
def save_projs(py_proj, c_proj):
    torch.save(py_proj.state_dict(), 'py_proj.pt')
    torch.save(c_proj.state_dict(), 'c_proj.pt')

save_projs(py_proj, c_proj)
# ------------------------------------------------------------------------------
try:
    temp = type(QUERY_EMBS)
    print('Here')
except:
    QUERY_EMBS, CAND_EMBS = None, None
    print('Not here')
# ------------------------------------------------------------------------------
def load_test_csv(path):
    queries = []
    candidates = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["type"] == "query":
                queries.append({"id": row["datapointID"], "source": row["source"]})
            elif row["type"] == "candidate":
                candidates.append({"id": row["datapointID"], "source": row["source"]})
    return queries, candidates

def solve(test_csv='test_data.csv', output_csv='submission.csv', batch_size=BATCH_SIZE, py_checkpoint='py_proj.pt', c_checkpoint='c_proj.pt'):
    global QUERY_EMBS, CAND_EMBS
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Loading tokenizer and model...")
    tokenizer, bert = load_tokenizer_and_model(MODEL_NAME)
    bert.eval()

    py_proj = None
    c_proj = None
    if py_checkpoint and c_checkpoint:
        print(f"Loading checkpoints: {py_checkpoint} and {c_checkpoint}")
        py_ckpt = torch.load(py_checkpoint, map_location=device, weights_only=True)
        c_ckpt = torch.load(c_checkpoint, map_location=device, weights_only=True)
        py_proj = ProjectionHead(EMBED_DIM, PROJ_DIM).to(device)
        c_proj = ProjectionHead(EMBED_DIM, PROJ_DIM).to(device)
        py_proj.load_state_dict(py_ckpt)
        c_proj.load_state_dict(c_ckpt)
        py_proj.eval()
        c_proj.eval()

    print(f"Loading test data from {test_csv}...")
    queries, candidates = load_test_csv(test_csv)
    print(f"Queries: {len(queries)}, Candidates: {len(candidates)}")

    query_texts = [q["source"] for q in queries]
    query_ids = [q["id"] for q in queries]
    cand_texts = [c["source"] for c in candidates]
    cand_ids = [c["id"] for c in candidates]

    if QUERY_EMBS is None:
        print("Encoding queries (Python)...")
        query_embs = encode_batch(bert, tokenizer, query_texts, device, batch_size)
        QUERY_EMBS = query_embs
    else:
        print(f'Caching queries (Python)...')
        query_embs = QUERY_EMBS
    print(f'query_embs shape: {query_embs.shape}')

    if CAND_EMBS is None:
        print("Encoding candidates (C++)...")
        cand_embs = encode_batch(bert, tokenizer, cand_texts, device, batch_size)
        CAND_EMBS = cand_embs
    else:
        print(f'Caching queries (C++)...')
        cand_embs = CAND_EMBS
    print(f'cand_embs shape: {cand_embs.shape}')

    if py_proj and c_proj:
        print("Applying projection heads...")
        with torch.no_grad():
            query_embs = py_proj(query_embs.to(device)).cpu()
            cand_embs = c_proj(cand_embs.to(device)).cpu()

    query_embs = F.normalize(query_embs, p=2, dim=-1)
    cand_embs = F.normalize(cand_embs, p=2, dim=-1)
    # cost_matrix = torch.matmul(query_embs, cand_embs.T)
    # row_ind, col_ind = hungarian(cost_matrix)

    print("Ranking candidates...")
    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL)
        writer.writerow(["subtaskID", "datapointID", "answer"])
        for i, qid in enumerate(tqdm(query_ids, desc="Scoring")):
            sims = torch.matmul(cand_embs, query_embs[i])
            ranked_indices = torch.argsort(sims, descending=True).tolist()
            ranked_ids = [cand_ids[j] for j in ranked_indices]
            writer.writerow([1, qid, ";".join(ranked_ids)])

    print(f"Submission saved to {output_csv}")

solve()
