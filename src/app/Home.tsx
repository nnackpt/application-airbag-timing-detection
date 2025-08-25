import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/Card";
import { getHealth, getVideos } from "@/lib/api";
import Link from "next/link";
import { JSX } from "react";

export default async function Home(): Promise<JSX.Element> {
    const [health, videos] = await Promise.all([getHealth(), getVideos(5)])
    const recent = videos ?? []
    
    return (
        <div className="space-y-8">
            <section className="relative overflow-hidden rounded-2xl bg-gradient-to-br from-white to-sky-50 border border-gray-100">
                <div className="p-8 md:p-12">
                    <h1 className="text-3xl md:text-4xl font-semibold tracking-tight mb-2">Airbag Timing Detection</h1>
                    <p className="text-gray-600 max-w-2xl">Upload videos and get a comprehensive analysis of the airbag deployment timing, helping you ensure optimal safety and performance.</p>
                    <div className="mt-6 flex gap-3">
                        <Link 
                            href="/upload" 
                            className="inline-flex items-center justify-center gap-2 rounded-2xl px-4 py-2 font-medium 
                            transition bg-[var(--primary-color)] text-white hover:opacity-90 active:opacity-80"
                            >
                                Upload Video
                        </Link>
                        <Link 
                            href="/history" 
                            className="inline-flex items-center justify-center gap-2 rounded-2xl px-4 py-2 font-medium 
                            transition border border-[var(--primary-color)] text-[var(--primary-color)] hover:bg-[var(--primary-color)]/10"
                            >
                                View History
                        </Link>
                    </div>
                </div>
            </section>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                <Card>
                    <CardHeader><CardTitle>System Health</CardTitle></CardHeader>
                    <CardContent>
                        <ul className="text-sm text-gray-700 space-y-1">
                            <li><span className="font-medium">Status:</span> {health.status}</li>
                            <li><span className="font-medium">Version:</span> {health.version}</li>
                            <li><span className="font-medium">Active tasks:</span> {health.storage.active_tasks}</li>
                        </ul>
                        <div className="mt-3 flex flex-wrap gap-2">
                            {Object.entries(health.models_loaded).map(([k, v]) => (
                                <span 
                                    key={k} 
                                    className={v 
                                        ? "inline-flex items-center rounded-full px-2.5 py-0.5 text-xs font-medium border bg-green-50 text-green-700 border-green-200" 
                                        : "inline-flex items-center rounded-full px-2.5 py-0.5 text-xs font-medium border badge bg-amber-50 text-amber-700 border-amber-200"
                                    }>{k}: {v ? "loaded" : "missing"}
                                </span>
                            ))}
                        </div>
                    </CardContent>
                </Card>

                <Card>
                    <CardHeader><CardTitle>Quick Links</CardTitle></CardHeader>
                    <CardContent>
                        <div className="flex flex-col gap-2 text-sm">
                            <Link className="text-[var(--primary-color)] hover:underline" href="/task">Open Task Status</Link>
                            <Link className="text-[var(--primary-color)] hover:underline" href="/screenshots">Open Screenshots</Link>
                            <Link className="text-[var(--primary-color)] hover:underline" href="/player">Open Player</Link>
                            <Link className="text-[var(--primary-color)] hover:underline" href="/health">Open Health</Link>
                        </div>
                    </CardContent>
                </Card>

                <Card>
                    <CardHeader><CardTitle>Enhancements</CardTitle></CardHeader>
                    <CardContent>
                        <ul className="list-disc list-inside text-sm text-gray-700 space-y-1">
                            {health.enhancements.slice(0, 6).map((e) => (<li key={e}>{e}</li>))}
                        </ul>
                    </CardContent>
                </Card>
            </div>

            <Card>
                <CardHeader><CardTitle>Recent Videos</CardTitle></CardHeader>
                <CardContent>
                    {recent.length === 0 ? (
                        <p className="text-gray-600">No videos yet.</p>
                    ) : (
                        <div className="overflow-x-auto">
                            <table className="min-w-full text-sm">
                                <thead>
                                    <tr className="text-left text-gray-500">
                                        <th className="py-2">Filename</th>
                                        <th className="py-2">Status</th>
                                        <th className="py-2">Progress</th>
                                        <th className="py-2">Temperature</th>
                                        <th className="py-2">Action</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {recent.map((v) => (
                                        <tr key={v.task_id} className="border-t border-gray-100">
                                            <td className="py-2">{v.video_filename}</td>
                                            <td className="py-2">
                                                <span className="inline-flex items-center rounded-full px-2.5 py-0.5 text-xs font-medium border 
                                                                badge bg-sky-50 text-sky-700 border-sky-200"
                                                >
                                                    {v.status}
                                                </span>
                                            </td>
                                            <td className="py-2">{v.progress}%</td>
                                            <td className="py-2">{v.temperature_type ?? "room"}</td>
                                            <td className="py-2">
                                                <Link 
                                                    className="inline-flex items-center justify-center gap-2 rounded-2xl px-4 py-2 font-medium 
                                                                transition border border-brand text-brand hover:bg-brand/10" 
                                                    href={`/task?taskId=${encodeURIComponent(v.task_id)}`}
                                                >
                                                    Open
                                                </Link>
                                            </td>
                                        </tr>
                                    ))}
                                </tbody>
                            </table>
                        </div>
                    )}
                </CardContent>
            </Card>
        </div>
    )
}