// import type { Metadata } from "next";
import './globals.css'
// import Link from "next/link";
import { Inter } from "next/font/google";
import Navigation from "@/components/Navigation";

const inter = Inter({ subsets: ['latin'] })

export const metadata = {
  title: "Airbag Timing Detection",
  description: "Advance AI-powered airbag deployment detection system"
};

export default function RootLayout({ 
  children 
}: { 
  children: React.ReactNode 
}) {
  return (
    <html lang="en" className="h-full">
      <body className={`${inter.className} h-full`}>
        <div className="min-h-screen bg-gradient-to-br from-gray-50 to-blue-50">
          <Navigation />
          <main className="pb-8">
            {children}
          </main>
        </div>
      </body>
    </html>
  )
}
