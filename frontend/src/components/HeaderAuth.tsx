'use client';

import { Show, SignInButton, SignUpButton, UserButton } from '@clerk/nextjs';

export default function HeaderAuth() {
  return (
    <div>
      <Show when="signed-out">
        <div className="flex items-center gap-3">
          <SignInButton mode="modal">
            <button className="text-sm font-medium text-slate-300 hover:text-white transition-colors">
              Sign In
            </button>
          </SignInButton>
          <SignUpButton mode="modal">
            <button className="px-4 py-2 bg-indigo-600 hover:bg-indigo-500 font-medium text-sm rounded-lg transition-colors text-white">
              Sign Up
            </button>
          </SignUpButton>
        </div>
      </Show>
      <Show when="signed-in">
        <UserButton />
      </Show>
    </div>
  );
}