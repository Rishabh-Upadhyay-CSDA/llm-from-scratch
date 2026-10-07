'use client';

import { Show, SignInButton, UserButton } from '@clerk/nextjs';

export default function HeaderAuth() {
  return (
    <div>
      <Show when="signed-out">
        <SignInButton mode="modal">
          <button className="px-4 py-2 bg-indigo-600 hover:bg-indigo-500 font-medium text-sm rounded-lg transition-colors text-white">
            Sign In
          </button>
        </SignInButton>
      </Show>
      <Show when="signed-in">
        <UserButton />
      </Show>
    </div>
  );
}