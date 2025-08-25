"use client"

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/Card"
import { getLogs } from "@/lib/api"
import { LogsResponse } from "@/lib/types"
import { fmtDate } from "@/lib/utils"
import { useSearchParams } from "next/navigation"
import { JSX, useEffect, useState } from "react"

export default function Logs(): JSX.Element {
    const sp = useSearchParams()
    const taskId = sp.get("taskId")
    const [data, setData] = useState<LogsResponse | null>(null)
    const [error, setError] = useState<string | null>(null)
 
    useEffect(() => {
        if (!taskId) return;
        (async () => {
            try {
                setData(await getLogs(taskId))
            } catch (e) {
                setError((e as Error).message)
            }
        })()
    }, [taskId])

    if (!taskId) 
        return (
            <Card>
                <CardContent>
                    <p className="text-gray-600">Provide a <span className="font-mono text-sm">taskId</span> via query string.</p>
                </CardContent>
            </Card>
        )

    return (
        <Card>
            <CardHeader>
                <CardTitle>
                    Processing Logs
                </CardTitle>
            </CardHeader>
            <CardContent>
                {!data ? (
                    <p className="text-gray-600">{error ?? "Loading..."}</p>
                ) : data.logs.length === 0 ? (
                    <p className="text-gray-600">No logs.</p>
                ) : (
                    <ul className="space-y-2 text-sm">
                        {data.logs.map((l, i) => (
                            <li key={i} className="grid grid-cols-12 gap-2 p-2 rounded-xl bg-gray-50">
                                <span className="col-span-3 font-mono text-sm">{fmtDate(l.timestamp)}</span>
                                <span className="col-span-2 uppercase text-gray-600">{l.log_level}</span>
                                <span className="col-span-7">{l.message}</span>
                            </li>
                        ))}
                    </ul>
                )}
            </CardContent>
        </Card>
    )
}