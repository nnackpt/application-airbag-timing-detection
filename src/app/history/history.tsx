import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/Card";
import { getVideos } from "@/lib/api";
import Link from "next/link";
import { JSX } from "react";

export default async function History(): Promise<JSX.Element> {
    const items = await getVideos(100)

    return (
        <Card>
            <CardHeader>
                <CardTitle>
                    Video History
                </CardTitle>
            </CardHeader>
            <CardContent>
                {items.length === 0 ? (
                    <p className="text-gray-600">Nothing here yet.</p>
                ) : (
                    <div className="overflow-x-auto">
                        <table className="min-w-full text-sm">
                            <thead>
                                <tr className="text-left text-gray-500">
                                    <th className="py-2">Filename</th>
                                    <th className="py-2">Status</th>
                                    <th className="py-2">Progress</th>
                                    <th className="py-2">Temperature</th>
                                    <th className="py-2">Created</th>
                                    <th className="py-2">Action</th>
                                </tr>
                            </thead>
                            <tbody>
                                {items.map((v) => (
                                    <tr key={v.task_id} className="border-t border-gray-100">
                                        <td className="py-2">{v.video_filename}</td>
                                        <td className="py-2">
                                            <span className="inline-flex items-center rounded-full px-2.5 py-0.5 text-xs font-medium border bg-sky-50 text-sky-700 border-sky-200">
                                                {v.status}    
                                            </span>
                                        </td>
                                        <td className="py-2">{v.progress}%</td>
                                        <td className="py-2">{v.temperature_type ?? "room"}</td>
                                        <td className="py-2">{new Date(v.created_at).toLocaleString()}</td>
                                        <td className="py-2">
                                            <Link
                                                className="inline-flex items-center justify-center gap-2 rounded-2xl px-4 py-2 font-medium transition border border-brand text-brand hover:bg-brand/10 mr-2"
                                                href={`/task?taskId=${encodeURIComponent(v.task_id)}`}
                                            >
                                                Task
                                            </Link>
                                            <Link
                                                className="inline-flex items-center justify-center gap-2 rounded-2xl px-4 py-2 font-medium transition border border-brand text-brand hover:bg-brand/10 mr-2"
                                                href={`/results?taskId=${encodeURIComponent(v.task_id)}`}
                                            >
                                                Results
                                            </Link><Link
                                                className="inline-flex items-center justify-center gap-2 rounded-2xl px-4 py-2 font-medium transition border border-brand text-brand hover:bg-brand/10"
                                                href={`/player?taskId=${encodeURIComponent(v.task_id)}`}
                                            >
                                                Player
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
    )
}