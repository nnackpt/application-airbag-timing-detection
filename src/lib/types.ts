export type TemperatureValue = "room" | "hot" | "cold"

export interface TemperatureOption {
    value: TemperatureValue
    label: string
    frame_range: string
}

export interface VideoUploadResponse {
    task_id: string
    message: string
    video_filename: string
    temperature_type: TemperatureValue
}

export interface ProcessingStatus {
    task_id: string
    status: "pending" | "processing" | "completed" | "failed"
    progress: number // 0-100
    message: string
    created_at: string
    completed_at: string | null
    temperature_type?: TemperatureValue
}

export interface VideoRecordItem {
    id: number
    task_id: string
    original_filename: string
    video_filename: string
    status: "pending" | "processing" | "completed" | "failed"
    progress: number
    message: string
    created_at: string
    completed_at: string | null
    output_video_path: string | null
    screenshots: string[]
    temperature_type?: TemperatureValue
}

export interface ScreenshotMeta {
    filename: string
    path: string
    full_path: string
}

export interface ScreenshotsResponse {
    screenshots: ScreenshotMeta[]
}

export interface OCRMap {
    [filename: string]: string
}

export interface DetectionResult {
    task_id: string
    video_filename: string
    explosion_frame: number | null
    full_deployment_frame: number | null
    detected_labels: string[]
    screenshots: string[]
    processing_time: number
    ocr_results: OCRMap
}

export interface HealthResponse {
    status: string
    version: string
    timestamp: string
    models_loaded: {
        yolo_object: boolean
        yolo_name: boolean
        sam: boolean
        ocr: boolean
    }
    storage: {
        type: string
        total_videos: number
        active_tasks: number
    }
    enhancements: string[]
}

export interface LogEntry {
    log_level: string
    message: string
    timestamp: string
}

export interface LogsResponse {
    task_id: string
    logs: LogEntry[]
}