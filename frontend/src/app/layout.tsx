import { ClerkProvider } from '@clerk/nextjs';
import HeaderAuth from '@/components/HeaderAuth';
import './globals.css';

export const dynamic = 'force-dynamic';

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <ClerkProvider>
      <html lang="en">
        <body className="bg-slate-900 text-slate-100 min-h-screen">
          <header className="flex justify-between items-center px-6 py-4 border-b border-slate-800 bg-slate-950">
            <h1 className="font-bold text-xl tracking-tight text-indigo-400">LLM Playground</h1>
            <HeaderAuth />
          </header>
          {children}
        </body>
      </html>
    </ClerkProvider>
  );
}