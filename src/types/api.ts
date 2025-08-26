// export type Temperature = "room" | "hot" | "cold"

export interface VideoUploadResponse {
    task_id: string
    message: string
    video_filename: string
    temperature_type: string
}

export interface ProcessingStatus {
    task_id: string
    status: "pending" | "processing" | "completed" | "failed"
    progress: number
    message: string
    created_at: string
    completed_at?: string
    temperature_type?: string
}

export interface VideoRecord {
    id: number
    task_id: string
    original_filename: string
    video_filename: string
    status: string
    progress: number
    message: string
    created_at: string
    completed_at?: string
    output_video_path?: string
    screenshots: string[]
    temperature_type?: string
}

export interface DetectionResult {
    task_id: string
    video_filename: string
    explosion_frame?: number
    full_deployment_frame?: number
    detected_labels: string[]
    screenshots: string[]
    processing_time: number
    ocr_results: Record<string, string>
}

export interface TemperatureOption {
    value: "room" | "hot" | "cold"
    label: string
    frame_range: string
}

export interface Screenshot {
    filename: string
    path: string
    full_path: string
}