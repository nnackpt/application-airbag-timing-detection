"use client"

import { SectionCard } from "@/components/SectionCard"
import { StepBadge } from "@/components/StepBadge"
import { VideoPlayer } from "@/components/VideoPlayer"
import { getScreenshots, getScreenshotUrl, getStatus, getVideoUrl, uploadVideo } from "@/lib/api"
import { ProcessingStatus, ScreenshotsResponse, Temperature } from "@/types/api"
import { useRouter, useSearchParams } from "next/navigation"
import { useCallback, useEffect, useMemo, useRef, useState } from "react"

const STEPS = [
    { key: "upload", label: "Upload Received" },
    { key: "circles", label: "Detecting Circles" },
    { key: "assign", label: "Matching Labels" },
    { key: "explosion", label: "Explosion Detection" },
    { key: "full", label: "Full Deployment Analysis" },
    { key: "ocr", label: "OCR on Screenshots" },
    { key: "done", label: "Completed" },
] as const

type StepKey = typeof STEPS[number]["key"]

function inferStepFormMessage(message: string, status: ProcessingStatus["status"]): StepKey {
    if (status === "completed") return "done"
    const m = message.toLowerCase()
    if (m.includes("ocr")) return "ocr"
    if (m.includes("full deployment")) return "full"
    if (m.includes("explosion")) return "explosion"
    if (m.includes("mapping") || m.includes("hungarian") || m.includes("finalize")) return "assign"
    if (m.includes("circle") || m.includes("consensus")) return "circles"
    if (m.includes("upload") || m.includes("opening video") || m.includes("queued")) return "upload"
    return "upload"
}

export default function Processing() {
    const router = useRouter()
    const params = useSearchParams()
    const existingTaskId = params.get("task_id")

    const [file, setFile] = useState<File | null>(null)
    const [temperature, setTemperature] = useState<Temperature>("room")
    const [taskId, setTaskId] = useState<string | null>(existingTaskId)
    const [status, setStatus] = useState<ProcessingStatus | null>(null)
    const [shots, setShots] = useState<ScreenshotsResponse | null>(null)
    const [error, setError] = useState<string | null>(null)
    const [isStarting, setIsStarting] = useState(false)
    const resultsRef = useRef<HTMLDivElement | null>(null)

    const currentStep = useMemo(() => {
        if (!status) return "upload" as StepKey
        return inferStepFormMessage(status.message, status.status)
    }, [status])

    const startPolling = useCallback((id: string) => {
        const interval = setInterval(async () => {
            try {
                const s = await getStatus(id)
                setStatus(s)
                if (s.status === "completed") {
                    clearInterval(interval)
                    const sc = await getScreenshots(id)
                    setShots(sc)
                    setTimeout(() => {
                       resultsRef.current?.scrollIntoView({ behavior: "smooth", block: "start" }) 
                    }, 200);
                } else if (s.status === "failed") {
                    clearInterval(interval)
                }
            } catch (e) {
                clearInterval(interval)
                setError("Failed to poll status")
            }
        }, 1500)
    }, [])

    useEffect(() => {
        if (existingTaskId) {
            startPolling(existingTaskId)
        }
    }, [existingTaskId, startPolling])

    const onStart = useCallback(async () => {
        setError(null)
        if (!file) {
            setError("Please choose a video file first.")
            return
        }
        try {
            setIsStarting(true)
            const resp = await uploadVideo(file, temperature)
            setTaskId(resp.task_id)
            const url = new URL(window.location.href)
            url.searchParams.set("task_id", resp.task_id)
            router.replace(url.pathname + "?" + url.searchParams.toString())
            startPolling(resp.task_id)
        } catch (e) {
            setError("Upload failed. See console for details.")
            console.error(e)
        } finally {
            setIsStarting(false)
        }
    }, [file, temperature, router, startPolling])

    const onReset = useCallback(() => {
        setFile(null)
        setTaskId(null)
        setStatus(null)
        setShots(null)
        setError(null)
        const url = new URL(window.location.href)
        url.searchParams.delete("task_id")
        router.replace(url.pathname)
        window.scrollTo({ top: 0, behavior: "smooth" })
    }, [router])

    return (
        <div className="space-y-6">
            <SectionCard title="Upload & Start">
                <div className="grid grid-cols-1 gap-4 md:grid-cols-3">
                    <div className="md:col-span-2">
                        <div className="flex flex-col gap-3">
                            <label className="text-sm font-medium text-gray-700">Video File</label>
                            <input 
                                type="file" 
                                accept=".mp4, .avi, .mov, .mkv"
                                onChange={(e) => setFile(e.target.files && e.target.files[0] ? e.target.files[0] : null)}
                                className="block w-full rounded-lg border border-gray-200 p-2.5 text-sm file:mr-4 file:rounded-lg
                                            file:border-0 file:bg-[#005496] file:px-4 file:py-2 file:text-sm file:font-semibold file:text-white
                                            hover:file:opacity-90"
                            />
                        </div>

                        <div className="mt-4">
                            <span className="text-sm font-medium text-gray-700">Temperature</span>
                            <div className="mt-2 inline-flex overflow-hidden rounded-xl border border-gray-200">
                                {(["room", "hot", "cold"] as Temperature[]).map((t) => (
                                    <button
                                        key={t}
                                        type="button"
                                        onClick={() => setTemperature(t)}
                                        className={[
                                            "px-4 py-2 text-sm font-medium",
                                            temperature === t ? "bg-[#005496] text-white" : "bg-white text-gray-700 hover:bg-gray-50"
                                        ].join(" ")}
                                    >
                                        {t === "room" ? "Room" : t === "hot" ? "Hot" : "Cold"}
                                    </button>
                                ))}
                            </div>
                        </div>
                        
                        <div className="mt-5 flex items-center gap-3">
                            <button
                                onClick={onStart}
                                disabled={isStarting}
                                className="inline-flex items-center gap-2 rounded-xl bg-[#005496] px-5 py-2.5 text-sm font-semibold text-white hover:opacity-90 disabled:opacity-50"
                            >
                                {isStarting ? "Starting..." : "Start Processing"}
                            </button>
                            <button
                                onClick={onReset}
                                className="rounded-xl border border-gray-200 bg-white px-4 py-2.5 text-sm font-medium text-gray-700 hover:bg-gray-50"
                            >
                                Upload New Clip
                            </button>
                        </div>

                        {error && <p className="mt-3 text-sm text-red-600">{error}</p>}
                    </div>

                    <div className="rounded-2xl border border-gray-100 p-4">
                        <h3 className="mb-3 text-sm font-semibold text-gray-700">Status</h3>
                        <div className="flex flex-col gap-3">
                            {STEPS.map((s, i) => {
                                const idx = STEPS.findIndex((st) => st.key === currentStep)
                                return (
                                    <StepBadge
                                        key={s.key}
                                        label={s.label}
                                        active={i === idx}
                                        done={i < idx || (status?.status === "completed" && s.key === "done")}
                                    />
                                )
                            })}
                        </div>

                        <div className="mt-4 rounded-lf bg-gray-50 p-3 text-sm">
                            <div className="mb-1 flex items-center justify-between">
                                <span className="font-medium text-gray-700">Progress</span>
                                <span className="text-gray-500">{status ? `${status.progress}%` : "0%"}</span>
                            </div>
                            <div className="h-2 w-full overflow-hidden rounded-full bg-gray-200">
                                <div 
                                    className="h-full bg-[#005496] transition-all"
                                    style={{ width: `${status?.progress ?? 0}%`}}
                                />
                            </div>
                            <p className="mt-2 text-gray-600">{status?.message ?? "Waiting to start..."}</p>
                        </div>
                    </div>
                </div>
            </SectionCard>

            <div ref={resultsRef} />

            <SectionCard title="Outputs" right={
                taskId ? (
                    <a
                        className="text-sm font-medium text-[#005496] hover:underline"
                        href={`?task_id=${taskId}`}
                    >Permalink</a>
                ) : null
            }>
                {!taskId && (
                    <p className="text-sm text-gray-500">Upload and start processing to see results here.</p>
                )}

                {taskId && status?.status === "completed" && (
                    <div className="space-y-6">
                        <div>
                            <h3 className="mb-3 text-sm font-semibold text-gray-700">Processed Video</h3>
                            <VideoPlayer src={getVideoUrl(taskId)} />
                        </div>

                        <div>
                            <h3 className="mb-3 text-sm font-semibold text-gray-700">Screenshots</h3>
                            {shots && shots.screenshots.length > 0 ? (
                                <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 md:grid-cols-3">
                                    {shots.screenshots.map((sc) => (
                                        <figure key={sc.filename} className="rounded-xl border border-gray-100 p-2">
                                            <img 
                                                src={sc.path ? getScreenshotUrl(sc.path) : ""}
                                                alt={sc.filename}
                                                className="h-auto w-full rounded-lg" 
                                            />
                                            <figcaption className="mt-2 truncate text-xs text-gray-500">{sc.filename}</figcaption>
                                        </figure>
                                    ))}
                                </div>
                            ) : (
                                <p className="text-sm text-gray-500">No screenshots available.</p>
                            )}
                        </div>
                    </div>
                )}

                {taskId && status && status.status !== "completed" && (
                    <p className="text-sm text-gray-500">Processing... Outputs will appear here automatically when done.</p>
                )}
            </SectionCard>
        </div>
    )
}