import { api, ApiError } from "@/lib/api"
import { TemperatureOption } from "@/types/api"
import { AlertCircle, Thermometer, Upload, Video } from "lucide-react"
import React, { useCallback, useRef, useState } from "react"
import LoadingSpinner from "./LoadingSpinner"

interface VideoUploaderProps {
    onUploadSuccess: (taskId: string, filename: string, temperatureType: string) => void
    onUploadError: (error: string) => void
}

export default function VideoUploader({ onUploadSuccess, onUploadError }: VideoUploaderProps) {
    const [isDragging, setIsDragging] = useState(false)
    const [selectedFile, setSelectedFile] = useState<File | null>(null)
    const [temperatureType, setTemperatureType] = useState<string>("room")
    const [temperatureOptions, setTemperatureOptions] = useState<TemperatureOption[]>([])
    const [isUploading, setIsUploading] = useState(false)
    const [error, setError] = useState<string | null>(null)
    const fileInputRef = useRef<HTMLInputElement>(null)

    useState(() => {
        api.getTemperatureOptions()
            .then(data => setTemperatureOptions(data.options))
            .catch(err => console.error("Failed to load temperature options:", err))
    })

    const handleDragOver = useCallback((e: React.DragEvent) => {
        e.preventDefault()
        setIsDragging(true)
    }, [])

    const handleDragLeave = useCallback((e: React.DragEvent) => {
        e.preventDefault()
        setIsDragging(false)
    }, [])

    const handleDrop = useCallback((e: React.DragEvent) => {
        e.preventDefault()
        setIsDragging(false)

        const files = Array.from(e.dataTransfer.files)
        if (files.length > 0) {
            handleFileSelection(files[0])
        }
    }, [])

    const handleFileSelection = (file: File) => {
        setError(null)

        // Validate type
        const validTypes = ['video/mp4', 'video/avi', 'video/mov', 'video/quicktime', 'video/x-msvideo']
        if (!validTypes.includes(file.type)) {
            setError("Please select a valid video file (MP4, AVI, MOV)")
            return
        }

        setSelectedFile(file)
    }

    const handleFileInput = (e: React.ChangeEvent<HTMLInputElement>) => {
        const file = e.target.files?.[0]
        if (file) {
            handleFileSelection(file)
        }
    }

    const handleUpload = async () => {
        if (!selectedFile) return

        setIsUploading(true)
        setError(null)

        try {
            const response = await api.uploadVideo(selectedFile, temperatureType)
            onUploadSuccess(response.task_id, response.video_filename, response.temperature_type)

            // reset
            setSelectedFile(null)
            setTemperatureType("room")
            if (fileInputRef.current) {
                fileInputRef.current.value = ""
            }
        } catch (err) {
            const errorMessage = err instanceof ApiError
                ? `Upload failed: ${err.message}`
                : 'Upload failed. Please try again.'
            setError(errorMessage)
            onUploadError(errorMessage)
        } finally {
            setIsUploading(false)
        }
    }

    const formatFileSize = (bytes: number): string => {
        if (bytes === 0) return '0 Bytes'
        const k = 1024
        const sizes = ['Bytes', 'KB', 'MB', 'GB']
        const i = Math.floor(Math.log(bytes) / Math.log(k))
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i]
    }

    return (
        <div className="bg-white rounded-xl shadow-lg p-8">
            <div className="text-center mb-8">
                <div className="mx-auto w-16 h-16 bg-[var(--primary-color)] rounded-full flex items-center justify-center mb-4">
                    <Video className="w-8 h-8 text-white" />
                </div>
                <h2 className="text-2xl font-bold text-gray-900 mb-2">Upload Video</h2>
                <p className="text-gray-600">Drag and drop your video file here or click to select a file.</p>
            </div>

            {/* File upload area */}
            <div
                className={`relative border-2 border-dashed rounded-lg p-8 transition-all duration-200 ${
                    isDragging
                        ? "border-[var(--primary-color)] bg-blue-50"
                        : selectedFile
                        ? "border-green-300 bg-green-50"
                        : "border-gray-300 hover:border-[var(--primary-color)] hover:bg-blue-50 cursor-pointer" 
                }`}
                onDragOver={handleDragOver}
                onDragLeave={handleDragLeave}
                onDrop={handleDrop}
            >
                <input 
                    ref={fileInputRef}
                    type="file" 
                    accept="video/*" 
                    onChange={handleFileInput}
                    className="absolute inset-0 w-full h-full opacity-0 cursor-pointer" 
                />

                <div className="text-center">
                    {selectedFile ? (
                        <div>
                            <div className="w-16 h-16 bg-green-100 rounded-full flex items-center justify-center mx-auto mb-4">
                                <Video className="w-8 h-8 text-green-600" />
                            </div>
                            <h3 className="text-lg font-semibold text-gray-900 mb-2">{selectedFile.name}</h3>
                            <p className="text-gray-500 mb-2">{formatFileSize(selectedFile.size)}</p>
                            <button
                                onClick={() => {
                                    setSelectedFile(null)
                                    if (fileInputRef.current) {
                                        fileInputRef.current.value = ""
                                    }
                                }}
                                className="text-red-600 hover:text-red-800 text-sm font-medium"
                            >
                                Remove
                            </button>
                        </div>
                    ) : (
                        <div>
                            <div className="w-16 h-16 bg-gray-100 rounded-full flex items-center justify-center mx-auto mb-4">
                                <Upload className="w-8 h-8 text-gray-400" />
                            </div>
                            <h3 className="text-lg font-semibold text-gray-900 mb-2">
                                Drag and drop your video here
                            </h3>
                            <p className="text-gray-500 mb-4">or click to browse</p>
                            <p className="text-xs text-gray-400">Support MP4, AVI, MOV files up to 100MB</p>
                        </div>
                    )}
                </div>
            </div>

            {/* Temperature select */}
            {selectedFile && (
                <div className="mt-6">
                    <label className="block text-sm font-semibold text-gray-900 mb-3">
                        <Thermometer className="w-4 h-4 inline mr-2" />
                        Temperature Condition
                    </label>
                    <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
                        {temperatureOptions.map((option) => (
                            <label key={option.value} className="cursor-pointer">
                                <input 
                                    type="radio"
                                    name="temperature"
                                    value={option.value}
                                    checked={temperatureType === option.value}
                                    onChange={(e) => setTemperatureType(e.target.value)}
                                    className="sr-only"
                                />
                                <div className={`p-4 rounded-lg border-2 transition-all duration-200 ${
                                    temperatureType === option.value
                                        ? "border-[var(--primary-color)] bg-blue-50 shadow-md"
                                        : "border-gray-200 hover:border-gray-300"
                                    }`}
                                >
                                    <div className="font-semibold text-gray-900">{option.label}</div>
                                    <div className="text-sm text-gray-600">Frames {option.frame_range}</div>
                                </div>
                            </label>
                        ))}
                    </div>
                </div>
            )}

            {/* Error display */}
            {error && (
                <div className="mt-6 p-4 bg-red-50 border border-red-200 rounded-lg flex items-center">
                    <AlertCircle className="w-5 h-5 text-red-600 mr-3 flex-shrink-0" />
                    <span className="text-red-800">{error}</span>
                </div>
            )}

            {/* Upload button */}
            {selectedFile && (
                <div className="mt-6 text-center">
                    <button
                        onClick={handleUpload}
                        disabled={isUploading}
                        className="bg-[var(--primary-color)] text-white px-8 py-3 rounded-lg font-semibold 
                                    hover:bg-[var(--primary-color-dark)] transition-all duration-200
                                    disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center mx-auto space-x-2"                   
                    >
                        {isUploading ? (
                            <>
                                <LoadingSpinner size="sm" className="text-white" />
                                <span>Start Analysis...</span>
                            </>
                        ) : (
                            <span>Start Detection Analysis</span>
                        )}
                    </button>
                </div>
            )}
        </div>
    )
}