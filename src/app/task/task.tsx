"use client"

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/Card"
import { Progress } from "@/components/ui/Progress"
import { getStatus, pollStatus } from "@/lib/api"
import { ProcessingStatus } from "@/lib/types"
import Link from "next/link"
import { useSearchParams } from "next/navigation"
import { useEffect, useState } from "react"

export default function Task() {
    const sp = useSearchParams()
    const taskId = sp.get("taskId")
    const [status, setStatus] = useState<ProcessingStatus | null>(null)
    const [error, setError] = useState<string | null>(null)
    const [busy, setBusy] = useState<boolean>(false)

    useEffect(() => {
        if (!taskId) return
        setBusy(true);
        (async () => {
            try {
                const s = await getStatus(taskId)
                setStatus(s)
                if (s.status !== "completed" && s.status !== "failed") {
                    const final = await pollStatus(taskId, setStatus)
                    setStatus(final)
                }
            } catch (e) {
                setError((e as Error).message)
            } finally {
                setBusy(false)
            }
        })()
    }, [taskId])

    if (!taskId) {
        return (
            <Card>
                <CardContent>
                    <p className="text-gray-600">
                        Provide a <span className="font-mono text-sm">taskId</span> via query string. Example: <span className="font-mono text-sm">/task?taskId=...</span>
                    </p>
                </CardContent>
            </Card>
        )
    }

    return (
        <div className="grid gap-6">
            <Card>
                <CardHeader>
                    <CardTitle>
                        Task Status
                    </CardTitle>
                </CardHeader>
                <CardContent>
                    <p className="text-sm text-gray-700">Task ID: <span className="font-mono text-sm">{taskId}</span></p>
                    {status ? (
                        <div className="mt-4 space-y-2">
                            <div className="flex items-center justify-between text-sm">
                                <span>Status</span>
                                <span className="inline-flex items-center rounded-full px-2.5 py-0.5 text-xs font-medium border">{status.status}</span>
                            </div>
                            <div className="flex items-center justify-between text-sm">
                                <span>Message</span>
                                <span className="text-gray-700">{status.message}</span>
                            </div>
                            <Progress value={status.progress} className="mt-2" />
                        </div>
                    ) : (
                        <p className="text-gray-600">{busy ? "Loading..." : (error ?? "No data yet.")}</p>
                    )}

                    <div className="mt-4 flex flex-wrap gap-2">
                        <Link
                            className="inline-flex items-center justify-center gap-2 rounded-2xl px-4 py-2 font-medium transition border border-brand text-brand hover:bg-brand/10"
                            href={`/results?taskId=${encodeURIComponent(taskId)}`}
                        >
                            Results
                        </Link>

                        <Link
                            className="inline-flex items-center justify-center gap-2 rounded-2xl px-4 py-2 font-medium transition border border-brand text-brand hover:bg-brand/10"
                            href={`/screenshots?taskId=${encodeURIComponent(taskId)}`}
                        >
                            Screenshots
                        </Link>

                        <Link
                            className="inline-flex items-center justify-center gap-2 rounded-2xl px-4 py-2 font-medium transition border border-brand text-brand hover:bg-brand/10"
                            href={`/player?taskId=${encodeURIComponent(taskId)}`}
                        >
                            Player
                        </Link>

                        <Link
                            className="inline-flex items-center justify-center gap-2 rounded-2xl px-4 py-2 font-medium transition border border-brand text-brand hover:bg-brand/10"
                            href={`/logs?taskId=${encodeURIComponent(taskId)}`}
                        >
                            Logs
                        </Link>
                    </div>
                </CardContent>
            </Card>
        </div>
    )
}