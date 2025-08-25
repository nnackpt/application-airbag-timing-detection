"use client"

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/Card"
import { getResults } from "@/lib/api"
import { DetectionResult } from "@/lib/types"
import { useSearchParams } from "next/navigation"
import { JSX, useEffect, useState } from "react"

export default function Results(): JSX.Element {
    const sp = useSearchParams()
    const taskId = sp.get("taskId")
    const [data, setData] = useState<DetectionResult | null>(null)
    const [error, setError] = useState<string | null>(null)

    useEffect(() => {
        if (!taskId) return;
        (async () => {
            try {
                setData(await getResults(taskId))
            } catch (e) {
                setError((e as Error).message)
            }
        })()
    }, [taskId])

    if (!taskId) {
        return (
            <Card>
                <CardContent>
                    <p className="text-gray-600">
                        Provide a <span className="font-mono text-sm">taskId</span> via query string.
                    </p>
                </CardContent>
            </Card>
        )
    }

    return (
        <Card>
            <CardHeader>
                <CardTitle>
                    Detection Results
                </CardTitle>
            </CardHeader>
            <CardContent>
                {data ? (
                    <div className="grid md:grid-cols-2 gap-6">
                        <div>
                            <ul className="text-sm text-gray-700 space-y-1">
                                <li><span className="font-medium">Video:</span> {data.video_filename}</li>
                                <li><span className="font-medium">Explosion frame:</span> {data.explosion_frame ?? "-"}</li>
                                <li><span className="font-medium">Full deployment frame:</span> {data.full_deployment_frame ?? "-"}</li>
                                <li><span className="font-medium">Labels:</span> {data.detected_labels.join(", ")}</li>
                                <li><span className="font-medium">Processing time:</span> {data.processing_time}s</li>
                            </ul>
                        </div>
                        <div>
                            <h4 className="text-xl md:text-2xl font-semibold mb-2">OCR</h4>
                            <div className="text-sm text-gray-700 space-y-1">
                                {Object.entries(data.ocr_results).length === 0 ? (
                                    <p>No OCR results.</p>
                                ) : (
                                    Object.entries(data.ocr_results).map(([k, v]) => (
                                        <div className="flex items-center justify-between" key={k}>
                                            <span className="font-mono truncate max-w-[60%]" title={k}>{k}</span>
                                            <span>{v}</span>
                                        </div>
                                    ))
                                )}
                            </div>
                        </div>
                    </div>
                ) : (
                    <p className="text-gray-600">{error ?? "Loading..."}</p>
                )}
            </CardContent>
        </Card>
    )
}