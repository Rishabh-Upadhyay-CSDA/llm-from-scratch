# Full-Stack LLM Playground

A full-stack application for serving, streaming, and interacting with custom PyTorch Transformer models built from scratch.

## Architecture

* **Frontend**: Next.js App Router (TypeScript), React, Tailwind CSS, `@clerk/nextjs` authentication
* **Backend**: FastAPI, PyTorch, SQLAlchemy, Uvicorn streaming server
* **Database**: Neon PostgreSQL
* **Deployment**: Vercel (Frontend), Render (Backend)

## Features

* **JWT Authentication**: Secured endpoints via Clerk JWKS token validation.
* **Real-time Streaming**: Server-Sent Events (SSE) streaming token output directly from PyTorch inference loops.
* **Prompt History**: Saved interaction histories powered by Neon PostgreSQL.
* **Modular Infrastructure**: Decoupled serverless client and cloud API service.

## Environment Setup

### 1. Backend (`.env`)

Create a `.env` file in the backend root directory:

```text
DATABASE_URL=postgresql://user:password@ep-example.neon.tech/neondb?sslmode=require
CLERK_JWKS_URL=https://example.clerk.accounts.dev/.well-known/jwks.json
MODEL_WEIGHTS_PATH=checkpoints/model.pt
```

### 2. Frontend (`.env.local`)

Create a `.env.local` file in the frontend root directory:

```text
NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY=pk_test_...
CLERK_SECRET_KEY=sk_test_...
NEXT_PUBLIC_API_BASE_URL=http://127.0.0.1:8000
NEXT_PUBLIC_CLERK_SIGN_IN_URL=/
NEXT_PUBLIC_CLERK_SIGN_IN_FALLBACK_REDIRECT_URL=/
NEXT_PUBLIC_CLERK_SIGN_OUT_REDIRECT_URL=/
```

## Quick Start

### Backend

1. Install dependencies:
   pip install -r requirements.txt

2. Run local FastAPI server:
   uvicorn api:app --host 0.0.0.0 --port 8000 --reload

### Frontend

1. Install dependencies:
   npm install

2. Run Next.js development server:
   npm run dev

Open http://localhost:3000 to view the application.

## License

Distributed under the MIT License. See LICENSE for more information.
