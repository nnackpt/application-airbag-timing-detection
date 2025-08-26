'use client'

import ProcessingHistory from "./history"

export default function HistoryPage() {
  return (
    <div className="min-h-screen">
      {/* Hero Section */}
      <div className="bg-gradient-to-r from-[var(--primary-color)] to-[var(--primary-dark)] text-white py-12">
        <div className="max-w-5xl mx-auto px-4 sm:px-6 lg:px-8">
            <ProcessingHistory />
        </div>
      </div>
    </div>
  )
}