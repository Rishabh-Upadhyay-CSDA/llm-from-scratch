import os
from dotenv import load_dotenv

load_dotenv()

import json
import asyncio
import jwt
import torch
import torch.nn.functional as F
from typing import Optional
from fastapi import FastAPI, Depends, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session
from tokenizers import Tokenizer

from database import engine, get_db, Base
from models import PromptHistory, ModelConfig
from model.transformer import LLMFromScratch

Base.metadata.create_all(bind=engine)

app = FastAPI(title="LLM Playground API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------------------------------------------------------
# 1. Clerk Authentication Setup
# -----------------------------------------------------------------------------
CLERK_JWKS_URL = os.getenv("CLERK_JWKS_URL")
jwks_client = jwt.PyJWKClient(CLERK_JWKS_URL)

def get_current_user_id(authorization: str = Header(...)) -> str:
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid authorization header format")
    token = authorization.split(" ")[1]
    try:
        signing_key = jwks_client.get_signing_key_from_jwt(token)

        payload = jwt.decode(token, signing_key, algorithms=["RS256"], options={"verify_aud": False})
        return payload["sub"]
    except Exception as e:
        raise HTTPException(status_code=401, detail=f"Invalid or expired token: {str(e)}")

# -----------------------------------------------------------------------------
# 2. Model & Tokenizer Initialization
# -----------------------------------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BLOCK_SIZE = ModelConfig.BLOCK_SIZE if hasattr(ModelConfig, "BLOCK_SIZE") else 128

tokenizer = Tokenizer.from_file("tokenizer/tokenizer.json")

VOCAB_SIZE = tokenizer.get_vocab_size()
D_MODEL = 128
N_LAYER = 4
N_HEAD = 4
BLOCK_SIZE = 128

model = LLMFromScratch(
    vocab_size=VOCAB_SIZE,
    d_model=D_MODEL,
    n_layer=N_LAYER,
    n_head=N_HEAD,
    block_size=BLOCK_SIZE
).to(DEVICE)

model.load_state_dict(torch.load("checkpoints/model.pt", map_location=DEVICE))
model.eval()

# Dummy model container check fallback if loading directly
if 'model' not in globals():
    raise RuntimeError("Ensure your trained PyTorch 'model' object is loaded into api.py!")

# -----------------------------------------------------------------------------
# 3. Streaming Text Generator with Top-K & Top-P Sampling
# -----------------------------------------------------------------------------
async def text_streamer(
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float
):
    encoded = tokenizer.encode(prompt)
    prompt_ids = encoded.ids if encoded.ids else [0]
    idx = torch.tensor([prompt_ids], dtype=torch.long, device=DEVICE)

    for i in range(max_new_tokens):
        idx_cond = idx[:, -BLOCK_SIZE:]

        with torch.no_grad():
            logits = model(idx_cond)

            if logits.dim() == 3:
                logits = logits[:, -1, :]

            # Mask out special/control tokens (<pad>, <unk>, <bos>, <eos>)
            logits[:, :4] = -float("Inf")

            # Temperature Scaling
            temp_val = max(temperature, 1e-5)
            logits = logits / temp_val

            # Top-K Filtering
            if top_k is not None and top_k > 0:
                top_k_val = min(top_k, logits.size(-1))
                indices_to_remove = logits < torch.topk(logits, top_k_val)[0][..., -1, None]
                logits[indices_to_remove] = -float("Inf")

            # Top-P (Nucleus) Filtering
            if top_p is not None and 0.0 < top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
                sorted_probabilities = F.softmax(sorted_logits, dim=-1)
                cumulative_probs = torch.cumsum(sorted_probabilities, dim=-1)

                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0

                indices_to_remove = sorted_indices_to_remove.scatter(
                    dim=-1, index=sorted_indices, src=sorted_indices_to_remove
                )
                logits[indices_to_remove] = -float("Inf")

            # Softmax & Sampling
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)

        idx = torch.cat((idx, idx_next), dim=1)

        # Token Decoding & Cleaning
        token_id = idx_next.item()
        decoded_text = tokenizer.decode([token_id], skip_special_tokens=True)

        if not decoded_text:
            raw_token = tokenizer.id_to_token(token_id) or ""
            decoded_text = raw_token.replace("Ġ", " ").replace(" ", " ")

        decoded_text = "".join(char for char in decoded_text if char.isprintable() or char == " ")

        payload = json.dumps({"token": decoded_text}) + "\n"
        yield payload.encode("utf-8")
        await asyncio.sleep(0.01)

# -----------------------------------------------------------------------------
# 4. API Request Schemas & Endpoints
# -----------------------------------------------------------------------------
class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 50
    temperature: float = 0.7
    top_k: int = 50
    top_p: float = 0.9

@app.post("/generate")
async def generate(
    req: GenerateRequest, 
    user_id: str = Depends(get_current_user_id),
    db: Session = Depends(get_db)
):
    full_response = ""

    async def wrapped_streamer():
        nonlocal full_response
        async for chunk in text_streamer(
            prompt=req.prompt,
            max_new_tokens=req.max_new_tokens,
            temperature=req.temperature,
            top_k=req.top_k,
            top_p=req.top_p
        ):
            # Capture streamed tokens to save complete text to Neon
            data = json.loads(chunk.decode("utf-8"))
            full_response += data.get("token", "")
            yield chunk

        # Persist entry to Neon Database after generation finishes
        history_entry = PromptHistory(
            user_id=user_id,
            prompt=req.prompt,
            response=full_response,
            temperature=str(req.temperature),
            top_k=req.top_k,
            top_p=str(req.top_p)
        )
        db.add(history_entry)
        db.commit()

    return StreamingResponse(wrapped_streamer(), media_type="application/x-ndjson")

@app.get("/history")
async def get_history(
    user_id: str = Depends(get_current_user_id),
    db: Session = Depends(get_db)
):
    records = (
        db.query(PromptHistory)
        .filter(PromptHistory.user_id == user_id)
        .order_by(PromptHistory.created_at.desc())
        .limit(50)
        .all()
    )
    return records