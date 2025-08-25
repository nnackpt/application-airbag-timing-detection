import type { Metadata } from "next";
// import { Geist, Geist_Mono } from "next/font/google";
import { JSX } from "react";
import './globals.css'
import Navbar from "@/components/layout/Navbar";
import Footer from "@/components/layout/Footer";

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

export const viewport = {
  themeColor: "#005496",
};

export default function RootLayout({ children }: { children: React.ReactNode }):JSX.Element {
  return (
    <html lang="en">
      <body>
          <Navbar />
          <main className="mx-auto max-w-6xl px-4 py-8">{children}</main>
          <Footer />
      </body>
    </html>
  )
}
