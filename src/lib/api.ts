import { DetectionResult, HealthResponse, LogsResponse, ProcessingStatus, ScreenshotsResponse, TemperatureOption, TemperatureValue, VideoRecordItem, VideoUploadResponse } from "./types"
import { API_BASE } from "./utils"

async function json<T>(res: Response): Promise<T> {
    if (!res.ok) {
        const text = await res.text()
        throw new Error(text || `HTTP ${res.status}`)
    }
    return res.json() as Promise<T>
}

export async function getTemperatureOptions(): Promise<TemperatureOption[]> {
    const res = await fetch(`${API_BASE}/temperature-options`, { cache: "no-store" })
    const data = (await json<{ options: TemperatureOption[] }>(res)).options
    return data
}

export async function uploadVideo(file: File, temperature: TemperatureValue): Promise<VideoUploadResponse> {
    const fd = new FormData()
    fd.append("file", file)
    fd.append("temperature_type", temperature)
    const res = await fetch(`${API_BASE}/upload`, { method: "POST", body: fd })
    return json<VideoUploadResponse>(res)
}

export async function getStatus(taskId: string): Promise<ProcessingStatus> {
    const res = await fetch(`${API_BASE}/status/${encodeURIComponent(taskId)}`, { cache: "no-store" })
    return json<ProcessingStatus>(res)
}

export async function pollStatus(taskId: string, onTick?: (s: ProcessingStatus) => void, intervalMs: number = 1500): Promise<ProcessingStatus> {
    for (;;) {
        const s = await getStatus(taskId)
        onTick?.(s)
        if (s.status === "completed" || s.status === "failed") return s
        await new Promise((r) => setTimeout(r, intervalMs))
    }
}

export async function getVideos(limit: number = 50): Promise<VideoRecordItem[]> {
    const res = await fetch(`${API_BASE}/videos?limit=${limit}`, { cache: "no-store" })
    return json<VideoRecordItem[]>(res)
}

export function videoStreamUrl(taskId: string): string {
    return `${API_BASE}/video/${encodeURIComponent(taskId)}`
}

export async function getScreenshots(taskId: string): Promise<ScreenshotsResponse> {
    const res = await fetch(`${API_BASE}/screenshots/${encodeURIComponent(taskId)}`, { cache: "no-store" })
    return json<ScreenshotsResponse>(res)
}

export function screenshotsUrl(taskId: string, filename: string): string {
    return `${API_BASE}/screenshots/${encodeURIComponent(taskId)}/${encodeURIComponent(filename)}`
}

export async function getResults(taskId: string): Promise<DetectionResult> {
    const res = await fetch(`${API_BASE}/results/${encodeURIComponent(taskId)}`, { cache: "no-store" })
    return json<DetectionResult>(res)
}

export async function getHealth(): Promise<HealthResponse> {
    const res = await fetch(`${API_BASE}/health`, { cache: "no-store" })
    return json<HealthResponse>(res)
}

export async function getLogs(taskId: string): Promise<LogsResponse> {
    const res = await fetch(`${API_BASE}/logs/${encodeURIComponent(taskId)}`, { cache: "no-store" })
    return json<LogsResponse>(res)
}

export async function deleteVideo(taskId: string): Promise<{ message: string }> {
    const res = await fetch(`${API_BASE}/video/${encodeURIComponent(taskId)}`, { method: "DELETE" })
    return json<{ message: string }>(res)
}