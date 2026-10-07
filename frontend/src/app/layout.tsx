import { ClerkProvider, SignInButton, SignedIn, SignedOut, UserButton } from '@clerk/nextjs';
import './globals.css';

export const dynamic = 'force-dynamic'; // Ensures Next.js does not attempt static prerendering for auth-dependent routes

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <ClerkProvider>
      <html lang="en">
        <body className="bg-slate-900 text-slate-100 min-h-screen">
          <header className="flex justify-between items-center px-6 py-4 border-b border-slate-800 bg-slate-950">
            <h1 className="font-bold text-xl tracking-tight text-indigo-400">LLM Playground</h1>
            <div>
              <SignedOut>
                <SignInButton mode="modal">
                  <button className="px-4 py-2 bg-indigo-600 hover:bg-indigo-500 font-medium text-sm rounded-lg transition-colors">
                    Sign In
                  </button>
                </SignInButton>
              </SignedOut>
              <SignedIn>
                <UserButton />
              </SignedIn>
            </div>
          </header>
          {children}
        </body>
      </html>
    </ClerkProvider>
  );
}