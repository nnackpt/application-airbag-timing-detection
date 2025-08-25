"use client"

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/Card";
import { getScreenshots, screenshotsUrl } from "@/lib/api";
import { ScreenshotsResponse } from "@/lib/types";
import { useSearchParams } from "next/navigation";
import { JSX, useEffect, useState } from "react";

export default function Screenshots(): JSX.Element {
    const sp = useSearchParams()
    const taskId = sp.get("taskId")
    const [data, setData] = useState<ScreenshotsResponse | null>(null)
    const [error, setError] = useState<string | null>(null)

    useEffect(() => {
        if (!taskId) return;
        (async () => {
            try {
                setData(await getScreenshots(taskId))
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
                    Screenshots
                </CardTitle>
            </CardHeader>
            <CardContent>
                {!data ? (
                    <p className="text-gray-600">{error ?? "Loading..."}</p>
                ) : data.screenshots.length === 0 ? (
                    <p className="text-gray-600">No screenshots found.</p>
                ) : (
                    <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4">
                        {data.screenshots.map((s) => (
                            <figure key={s.filename} className="space-y-2">
                                <img
                                    loading="lazy" 
                                    src={screenshotsUrl(taskId, s.filename)} 
                                    alt={s.filename}
                                    className="w-full rounded-xl border border-gray-200" 
                                />
                                <figcaption className="font-mono text-xs truncate" title={s.filename}>{s.filename}</figcaption>
                            </figure>
                        ))}
                    </div>
                )}
            </CardContent>
        </Card>
    )
}