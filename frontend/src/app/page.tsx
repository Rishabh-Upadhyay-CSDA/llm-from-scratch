'use client';

export const dynamic = 'force-dynamic';

import { useAuth, SignedIn, SignedOut, SignInButton, UserButton } from '@clerk/nextjs';
import { useState, useEffect } from 'react';

interface HistoryItem {
  id: number;
  prompt: string;
  response: string;
  temperature: string;
  created_at: string;
}

export default function Home() {
  const { getToken, isSignedIn, isLoaded } = useAuth();
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
      setMounted(true);
    }, []);

    if (!mounted || !isLoaded) {
      return (
        <div className="flex min-h-screen items-center justify-center bg-slate-900 text-slate-400">
          Loading LLM Playground...
        </div>
      );
    }

  const [prompt, setPrompt] = useState('');
  const [generation, setGeneration] = useState('');
  const [loading, setLoading] = useState(false);
  const [history, setHistory] = useState<HistoryItem[]>([]);
  
  // Model hyperparameters
  const [temperature, setTemperature] = useState(0.7);
  const [topK, setTopK] = useState(50);
  const [topP, setTopP] = useState(0.9);
  const [maxTokens, setMaxTokens] = useState(60);

  const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL || "http://127.0.0.1:8000";

  const fetchHistory = async () => {
    if (!isSignedIn) return;
    try {
      const token = await getToken();
      const res = await fetch(`${API_BASE_URL}/history`, {
        headers: { Authorization: `Bearer ${token}` }
      });
      if (res.ok) {
        const data = await res.json();
        setHistory(data);
      }
    } catch (err) {
      console.error('Failed to fetch history:', err);
    }
  };

  useEffect(() => {
    fetchHistory();
  }, [isSignedIn]);

  const handleGenerate = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!prompt.trim() || loading) return;

    setLoading(true);
    setGeneration('');

    try {
      const token = await getToken();
      const response = await fetch(`${API_BASE_URL}/generate`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${token}`
        },
        body: JSON.stringify({
          prompt,
          max_new_tokens: maxTokens,
          temperature,
          top_k: topK,
          top_p: topP
        })
      });

      if (!response.body) return;

      const reader = response.body.getReader();
      const decoder = new TextDecoder();

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;

        const chunk = decoder.decode(value, { stream: true });
        const lines = chunk.split('\n').filter((l) => l.trim() !== '');

        for (const line of lines) {
          try {
            const parsed = JSON.parse(line);
            if (parsed.token) {
              setGeneration((prev) => prev + parsed.token);
            }
          } catch (e) {
            // Ignore partial NDJSON chunks
          }
        }
      }

      // Refresh history list after stream completes
      await fetchHistory();
    } catch (err) {
      console.error('Error during generation stream:', err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex h-[calc(100vh-65px)]">
      <SignedOut>
        <div className="flex-1 flex flex-col items-center justify-center p-6 text-center">
          <h2 className="text-2xl font-bold mb-2">Welcome to LLM Playground</h2>
          <p className="text-slate-400 mb-6">Please sign in to access generation parameters and save prompt history.</p>
          <SignInButton mode="modal">
            <button className="px-6 py-3 bg-indigo-600 hover:bg-indigo-500 font-semibold rounded-lg">
              Sign In to Continue
            </button>
          </SignInButton>
        </div>
      </SignedOut>

      <SignedIn>
        {/* Sidebar History Column */}
        <aside className="w-80 border-r border-slate-800 bg-slate-950 p-4 overflow-y-auto flex flex-col">
          <h2 className="font-semibold text-slate-300 text-sm mb-4 uppercase tracking-wider">Prompt History</h2>
          <div className="space-y-3 flex-1">
            {history.length === 0 ? (
              <p className="text-xs text-slate-500">No previous prompts saved in Neon.</p>
            ) : (
              history.map((item) => (
                <div 
                  key={item.id} 
                  onClick={() => { setPrompt(item.prompt); setGeneration(item.response); }}
                  className="p-3 bg-slate-900 border border-slate-800 hover:border-indigo-500 rounded-lg cursor-pointer transition-colors group"
                >
                  <p className="font-medium text-sm text-slate-200 truncate group-hover:text-indigo-400">{item.prompt}</p>
                  <p className="text-xs text-slate-500 truncate mt-1">{item.response}</p>
                </div>
              ))
            )}
          </div>
        </aside>

        {/* Main Interface */}
        <main className="flex-1 flex flex-col p-6 space-y-6 overflow-y-auto">
          {/* Output Display */}
          <div className="flex-1 bg-slate-950 border border-slate-800 rounded-xl p-5 flex flex-col font-mono text-sm leading-relaxed">
            <span className="text-xs text-slate-500 font-sans mb-2">OUTPUT STREAM:</span>
            <div className="flex-1 whitespace-pre-wrap text-slate-200">
              {generation || <span className="text-slate-600">Model output will stream here...</span>}
            </div>
          </div>

          {/* Hyperparameters Controls */}
          <div className="grid grid-cols-4 gap-4 bg-slate-950 p-4 border border-slate-800 rounded-xl text-xs">
            <div>
              <label className="block text-slate-400 mb-1">Temperature: {temperature}</label>
              <input type="range" min="0.1" max="1.5" step="0.1" value={temperature} onChange={(e) => setTemperature(parseFloat(e.target.value))} className="w-full" />
            </div>
            <div>
              <label className="block text-slate-400 mb-1">Top-P: {topP}</label>
              <input type="range" min="0.1" max="1.0" step="0.05" value={topP} onChange={(e) => setTopP(parseFloat(e.target.value))} className="w-full" />
            </div>
            <div>
              <label className="block text-slate-400 mb-1">Top-K: {topK}</label>
              <input type="number" min="1" max="100" value={topK} onChange={(e) => setTopK(parseInt(e.target.value))} className="w-full bg-slate-900 border border-slate-800 p-1 rounded text-slate-200" />
            </div>
            <div>
              <label className="block text-slate-400 mb-1">Max Tokens: {maxTokens}</label>
              <input type="number" min="10" max="250" value={maxTokens} onChange={(e) => setMaxTokens(parseInt(e.target.value))} className="w-full bg-slate-900 border border-slate-800 p-1 rounded text-slate-200" />
            </div>
          </div>

          {/* Input Form */}
          <form onSubmit={handleGenerate} className="flex gap-3">
            <input
              type="text"
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              placeholder="Type your prompt..."
              className="flex-1 bg-slate-950 border border-slate-800 rounded-xl px-4 py-3 text-sm focus:outline-none focus:border-indigo-500 text-slate-100"
            />
            <button
              type="submit"
              disabled={loading}
              className="px-6 py-3 bg-indigo-600 hover:bg-indigo-500 disabled:bg-slate-800 text-slate-100 font-medium text-sm rounded-xl transition-colors"
            >
              {loading ? 'Generating...' : 'Generate'}
            </button>
          </form>
        </main>
      </SignedIn>
    </div>
  );
}