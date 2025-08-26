export type Temperature = "room" | "hot" | "cold"

export interface VideoUploadResponse {
    task_id: string
    message: string
    video_filename: string
    temperature_type: Temperature
}

export interface ProcessingStatus {
    task_id: string
    status: "pending" | "processing" | "completed" | "failed"
    progress: number
    message: string
    created_at: string
    completed_at?: string | null
    temperature_type?: Temperature
}

export interface ScreenshotInfo {
    filename: string
    path: string
    full_path: string
}

export interface ScreenshotsResponse {
    screenshots: ScreenshotInfo[]
}

export interface DetectionResult {
    task_id: string
    video_filename: string
    explosion_frame: number | null
    full_deployment_frame: number | null
    detected_labels: string[]
    screenshots: string[]
    processing_time: number
    ocr_results: Record<string, string>
}

export interface VideoRecord {
    id: number
    task_id: string
    original_filename: string
    video_filename: string
    status: "pending" | "processing" | "completed" | "failed"
    progress: number
    message: string
    created_at: string
    completed_at?: string | null
    output_video_path?: string | null
    screenshots: string[]
    temperature_type?: Temperature
}