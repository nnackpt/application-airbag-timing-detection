import Link from "next/link";
import { JSX } from "react";

export default function Navbar(): JSX.Element {
    return (
      <header className="sticky top-0 z-40 w-full border-b border-gray-100 bg-white/80 backdrop-blur supports-[backdrop-filter]:bg-white/70">
        <div className="mx-auto max-w-6xl px-4">
          <div className="flex h-16 items-center justify-between">
            <Link href="/" className="flex items-center gap-2">
              <div className="h-8 w-8 rounded-xl bg-[var(--primary-color)]" />
              <span className="font-semibold">Airbag Timing Detection</span>
            </Link>
            <nav className="hidden md:flex items-center gap-4 itext-sm">
                <Link href="/upload" className="text-gray-700 hover:text-[var(--primary-color)]">Upload</Link>
                <Link href="/task" className="text-gray-700 hover:text-[var(--primary-color)]">Task</Link>
                <Link href="/history" className="text-gray-700 hover:text-[var(--primary-color)]">History</Link>
                <Link href="/screenshots" className="text-gray-700 hover:text-[var(--primary-color)]">Screenshots</Link>
                <Link href="/player" className="text-gray-700 hover:text-[var(--primary-color)]">Player</Link>
                <Link href="/health" className="text-gray-700 hover:text-[var(--primary-color)]">Health</Link>
            </nav>
            <Link 
              href="/upload" 
              className="inline-flex items-center justify-center gap-2 rounded-2xl px-4 py-2 font-medium transition 
                        bg-[var(--primary-color)] text-white hover:opacity-90 active:opacity-80"
              >
                New Task
              </Link>
          </div>
        </div>
      </header>
    );
}