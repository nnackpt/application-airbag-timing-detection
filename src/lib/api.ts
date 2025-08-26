import { DetectionResult, ProcessingStatus, ScreenshotsResponse, Temperature, VideoRecord, VideoUploadResponse } from "@/types/api";

const API_BASE = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000"

export async function uploadVideo(file: File, temperature: Temperature): Promise<VideoUploadResponse> {
    const form = new FormData()
    form.append("file", file)
    form.append("temperature_type", temperature)

    const res = await fetch(`${API_BASE}/upload`, {
        method: "POST",
        body: form
    })
    if (!res.ok) throw new Error(await res.text())
        return (await res.json()) as VideoUploadResponse
}

export async function getStatus(taskId: string): Promise<ProcessingStatus> {
    const res = await fetch(`${API_BASE}/status/${taskId}`, { cache: "no-store" })
    if (!res.ok) throw new Error(await res.text())
    return (await res.json()) as ProcessingStatus
}

export async function getScreenshots(taskId: string): Promise<ScreenshotsResponse> {
    const res = await fetch(`${API_BASE}/screenshots/${taskId}`, { cache: "no-store" })
    if (!res.ok) throw new Error(await res.text())
    return (await res.json()) as ScreenshotsResponse
}

export function getVideoUrl(taskId: string): string {
    return `${API_BASE}/video/${taskId}`
}

export async function getResults(taskId: string): Promise<DetectionResult> {
    const res = await fetch(`${API_BASE}/results/${taskId}`, { cache: "no-store" })
    if (!res.ok) throw new Error(await res.text())
    return (await res.json()) as DetectionResult
}

export async function listVideos(limit = 50): Promise<VideoRecord[]> {
    const url = new URL(`${API_BASE}/videos`)
    url.searchParams.set("limit", String(limit))
    const res = await fetch(url.toString(), { cache: "no-store" })
    if (!res.ok) throw new Error(await res.text())
    return (await res.json()) as VideoRecord[]
}

export function getScreenshotUrl(path: string): string {
    return `${API_BASE}${path}`
}