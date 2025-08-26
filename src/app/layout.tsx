import type { Metadata } from "next";
// import { Geist, Geist_Mono } from "next/font/google";
import './globals.css'
import Link from "next/link";

// const geistSans = Geist({
//   variable: "--font-geist-sans",
//   subsets: ["latin"],
// });

// const geistMono = Geist_Mono({
//   variable: "--font-geist-mono",
//   subsets: ["latin"],
// });

export const metadata: Metadata = {
  title: "Airbag Timing Detection",
  description: "Modern dashboard for The Airbag Timing Detection"
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="bg-white text-gray-900 antialiased">
        <header className="sticky top-0 z-30 border-b border-gray-100 bg-white/90 backdrop-blur">
          <div className="mx-auto flex max-w-6xl items-center justify-between px-4 py-3">
            <div className="flex items-center gap-2">
              {/* <div className="h-8 w-8 rounded bg-[#005496]" /> */}
              <span className="text-xl font-bold text-[#005496]">Airbag Timing Detection</span>
            </div>
            <nav className="flex items-center gap-6 text-sm">
              <Link 
                className="font-medium text-gray-700 hover:text-[#005496]"
                href="/processing"
              >
                Processing
              </Link>
              <Link
                className="font-medium text-gray-700 hover:text-[#005496]"
                href="/history"
              >
                History
              </Link>
            </nav>
          </div>
        </header>
        <main className="mx-auto max-w-6xl px-4 py-8">{children}</main>
        <footer className="border-t border-gray-100 py-6">
          <div className="mx-auto max-w-6xl px-4 text-sm text-gray-500">© {new Date().getFullYear()} Airbag Timing Detection.</div>
        </footer>
      </body>
    </html>
  )
}
