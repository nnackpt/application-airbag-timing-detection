"use client"

import Button from "@/components/ui/Button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/Card";
import { getTemperatureOptions, uploadVideo } from "@/lib/api";
import { TemperatureOption, TemperatureValue, VideoUploadResponse } from "@/lib/types";
import Link from "next/link";
import { JSX, useEffect, useState } from "react";

export default function Upload(): JSX.Element {
    const [file, setFile] = useState<File | null>(null)
    const [opts, setOpts] = useState<TemperatureOption[]>([])
    const [temp, setTemp] = useState<TemperatureValue>("room")
    const [loading, setLoading] = useState<boolean>(false)
    const [resp, setResp] = useState<VideoUploadResponse | null>(null)
    const [error, setError] = useState<string | null>(null)

    useEffect(() => {
        void (async () => setOpts(await getTemperatureOptions()))()
    }, [])

    async function onSubmit(): Promise<void> {
        setError(null)
        if (!file) { setError("Please choose a video file (.mp4, .avi, .mov, .mkv)"); return }
        try {
            setLoading(true)
            const r = await uploadVideo(file, temp)
            setResp(r)
        } catch (e) {
            setError((e as Error).message)
        } finally {
            setLoading(false)
        }
    }

    return (
        <div className="grid gap-6">
            <Card>
                <CardHeader>
                    <CardTitle>
                        Upload Video
                    </CardTitle>
                </CardHeader>
                <CardContent>
                    <div className="grid gap-4">
                        <label className="block">
                            <span className="block text-sm font-medium text-gray-700">Video file</span>
                            <input 
                                type="file" 
                                accept="video/mp4,video/avi,video/quicktime,video/x-matroska"
                                onChange={(e) => setFile(e.currentTarget.files?.[0] ?? null)}
                                className="mt-1 block w-full rounded-xl border border-gray-200 px-3 py-2"
                            />
                        </label>

                        <label className="block">
                            <span className="block text-sm font-medium text-gray-700">Temperature</span>
                            <select 
                                value={temp}
                                onChange={(e) => setTemp(e.currentTarget.value as TemperatureValue)}
                                className="mt-1 block w-full rounded-xl border border-gray-200 px-3 py-2 bg-white"
                            >
                                {opts.map((o) => (
                                    <option key={o.value} value={o.value}>{o.label} ({o.frame_range})</option>
                                ))}
                            </select>
                        </label>

                        <div className="flex items-center gap-3">
                            <Button 
                                onClick={() => void onSubmit()}
                                disabled={loading}
                            >
                                {loading ? "Uploading..." : "Start Processing"}
                            </Button>
                            <Link
                                className="inline-flex items-center justify-center gap-2 rounded-2xl px-4 py-2 font-medium transition 
                                border border-[var(--primary-color)] text-[var(--primary-color)] hover:bg-[var(--primary-color)]/10"
                                href="/task"
                            >
                                Go to Task Status
                            </Link>
                        </div>

                        {error ? <p className="text-red-600 text-sm">{error}</p> : null}
                    </div>
                </CardContent>
            </Card>

            {resp ? (
                <Card>
                    <CardHeader>
                        <CardTitle>
                            Upload Successful
                        </CardTitle>
                    </CardHeader>
                    <CardContent>
                        <p className="text-sm text-gray-700">Task ID: <span className="font-mono text-sm">{resp.task_id}</span></p>
                        <div className="mt-3 flex gap-3">
                            <Link
                                className="inline-flex items-center justify-center gap-2 rounded-2xl px-4 py-2 font-medium 
                                transition bg-[var(--primary-color)] text-white hover:opacity-90 active:opacity-80"
                                href={`/task?taskId=${encodeURIComponent(resp.task_id)}`}
                            >
                                Open Task
                            </Link>

                            <Link
                                className="inline-flex items-center justify-center gap-2 rounded-2xl px-4 py-2 font-medium 
                                transition border border-[var(--primary-color)] text-[var(--primary-color)] hover:bg-[var(--primary-color)]/10"
                                href={`/player?taskId=${encodeURIComponent(resp.task_id)}`}
                            >
                                Open Player
                            </Link>
                        </div>
                    </CardContent>
                </Card>
            ) : null}
        </div >
    )
}