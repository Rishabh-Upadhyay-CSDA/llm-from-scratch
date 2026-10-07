'use client';

export const dynamic = 'force-dynamic';

import { useAuth, SignedIn, SignedOut, SignInButton } from '@clerk/nextjs';
import { useState, useEffect } from 'react';

export default function Home() {
  const { getToken, isSignedIn, isLoaded } = useAuth();
  const [mounted, setMounted] = useState(false);
  const [prompt, setPrompt] = useState('');
  const [generation, setGeneration] = useState('');
  const [loading, setLoading] = useState(false);

  const API_BASE = process.env.NEXT_PUBLIC_API_BASE_URL || 'http://127.0.0.1:8000';

  useEffect(() => {
    setMounted(true);
  }, []);

  if (!mounted || !isLoaded) {
    return (
      <div className="flex min-h-screen items-center justify-center bg-slate-900 text-slate-400 font-mono">
        Loading Playground...
      </div>
    );
  }

  const handleGenerate = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!prompt.trim() || loading) return;

    setLoading(true);
    setGeneration('');

    try {
      const token = await getToken();
      const res = await fetch(`${API_BASE}/generate`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${token}`
        },
        body: JSON.stringify({ prompt, max_new_tokens: 50, temperature: 0.7, top_k: 50, top_p: 0.9 })
      });

      if (!res.ok) {
        const errText = await res.text();
        setGeneration(`API Error (${res.status}): ${errText}`);
        setLoading(false);
        return;
      }

      const reader = res.body?.getReader();
      const decoder = new TextDecoder();

      while (reader) {
        const { value, done } = await reader.read();
        if (done) break;
        const chunk = decoder.decode(value, { stream: true });
        const lines = chunk.split('\n').filter((l) => l.trim() !== '');
        for (const line of lines) {
          try {
            const parsed = JSON.parse(line);
            if (parsed.token) setGeneration((prev) => prev + parsed.token);
          } catch (e) {}
        }
      }
    } catch (err: any) {
      setGeneration(`Stream network error: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };

  return (
    <main className="p-6 max-w-4xl mx-auto space-y-4">
      <SignedOut>
        <div className="text-center py-10">
          <p className="mb-4 text-slate-300">Please sign in to access the LLM generator.</p>
          <SignInButton mode="modal">
            <button className="px-4 py-2 bg-indigo-600 hover:bg-indigo-500 font-medium text-sm rounded-lg transition-colors text-white">
              Sign In
            </button>
          </SignInButton>
        </div>
      </SignedOut>

      <SignedIn>
        <div className="p-4 bg-slate-950 border border-slate-800 rounded font-mono min-h-[150px] whitespace-pre-wrap">
          {generation || 'Output will stream here...'}
        </div>

        <form onSubmit={handleGenerate} className="flex gap-2">
          <input
            type="text"
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            placeholder="Type prompt..."
            className="flex-1 bg-slate-950 border border-slate-800 p-2 rounded text-white"
          />
          <button type="submit" disabled={loading} className="px-4 py-2 bg-indigo-600 rounded text-white font-medium">
            {loading ? 'Streaming...' : 'Generate'}
          </button>
        </form>
      </SignedIn>
    </main>
  );
}