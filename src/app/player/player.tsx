"use client"

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/Card";
import { videoStreamUrl } from "@/lib/api";
import { useSearchParams } from "next/navigation";
import { JSX } from "react";

export default function Player(): JSX.Element {
    const sp = useSearchParams()
    const taskId = sp.get("taskId")

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
                    Video Player
                </CardTitle>
            </CardHeader>
            <CardContent>
                <video 
                    controls
                    className="w-full rounded-2xl border border-gray-200"
                    src={videoStreamUrl(taskId)}
                />
            </CardContent>
        </Card>
    )
}