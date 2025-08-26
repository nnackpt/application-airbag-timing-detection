"use client"

import { listVideos } from "@/lib/api";
import { VideoRecord } from "@/types/api";
import Link from "next/link";
import { useEffect, useMemo, useState } from "react";

export default function History() {
    const [items, setItems] = useState<VideoRecord[]>([])
    const [q, setQ] = useState<string>("")
    const [loading, setLoading] = useState<boolean>(true)
    const [error, setError] = useState<string | null>(null)
    
    useEffect(() => {
        (async () => {
            try {
                const data = await listVideos(50)
                setItems(data)
            } catch (e) {
                setError("Failed to load history.")
                console.error(e)
            } finally {
                setLoading(false)
            }
        })()
    }, [])

    const filtered = useMemo(() => {
        const qq = q.trim().toLowerCase()
        if (!qq) return items
        return items.filter((i) =>
            i.video_filename.toLowerCase().includes(qq) || 
            i.status.toLowerCase().includes(qq) ||
            (i.temperature_type ?? "").toLowerCase().includes(qq)
        )
    }, [items, q])

    return (
        <div className="space-y-6">
            <div className="flex flex-col items-start justify-between gap-3 sm:flex-row sm:items-center">
                <h1 className="text-2xl font-bold text-gray-900">History</h1>
                <input 
                    value={q}
                    onChange={(e) => setQ(e.target.value)}
                    placeholder="Search by filename, status, or temperature..."
                    className="w-full rounded-xl border border-gray-200 px-4 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-[#005496] sm:w-96"
                />
            </div>

            {loading && <p className="text-sm text-gray-500">Loading...</p>}
            {error && <p className="text-sm text-red-600">{error}</p>}

            {!loading && filtered.length === 0 && (
                <p className="text-sm text-gray-500">No records.</p>
            )}

            {!loading && filtered.length > 0 && (
                <div className="overflow-hidden rounded-2xl border border-gray-100">
                    <table className="min-w-full divide-y divide-gray-100">
                        <thead className="bg-gray-50">
                            <tr>
                                <th className="px-4 py-3 text-left text-xs font-semibold uppercase tracking-wide text-gray-500">Time</th>
                                <th className="px-4 py-3 text-left text-xs font-semibold uppercase tracking-wide text-gray-500">File</th>
                                <th className="px-4 py-3 text-left text-xs font-semibold uppercase tracking-wide text-gray-500">Temperature</th>
                                <th className="px-4 py-3 text-left text-xs font-semibold uppercase tracking-wide text-gray-500">Status</th>
                                <th className="px-4 py-3 text-left text-xs font-semibold uppercase tracking-wide text-gray-500">Progress</th>
                                <th className="px-4 py-3 text-left text-xs font-semibold uppercase tracking-wide text-gray-500">Action</th>
                            </tr>
                        </thead>
                        <tbody className="divide-y divide-gray-100 bg-white">
                            {filtered.map((r) => (
                                <tr key={r.task_id} className="hover:bg-gray-50">
                                    <td className="px-4 py-3 text-sm text-gray-600">{new Date(r.created_at).toLocaleString()}</td>
                                    <td className="px-4 py-3 text-sm font-medium text-gray-800">{r.video_filename}</td>
                                    <td className="px-4 py-3 text-sm text-gray-700">{r.temperature_type ?? "room"}</td>
                                    <td className="px-4 py-3">
                                        <span 
                                            className={[
                                                "inline-flex items-center rounded-full px-2.5 py-1 text-xs font-semibold",
                                                r.status === "completed"
                                                    ? "bg-green-100 text-green-800"
                                                    : r.status === "failed"
                                                    ? "bg-red-100 text-red-800"
                                                    : "bg-gray-100 text-gray-800"  
                                            ].join(" ")}
                                        >
                                            {r.status}
                                        </span>
                                    </td>
                                    <td className="px-4 py-3 text-sm text-gray-700">{r.progress}%</td>
                                    <td className="px-4 py-3 text-sm">
                                        <Link
                                            href={`/processing?task_id=${encodeURIComponent(r.task_id)}`}
                                            className="text-[#005496] hover:underline"
                                        >
                                            View
                                        </Link>
                                    </td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                </div>
            )}
        </div>
    )
}