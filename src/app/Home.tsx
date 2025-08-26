import ProcessingProgress from "@/components/ProcessingProgress"
import ResultsDisplay from "@/components/ResultsDisplay"
import VideoUploader from "@/components/VideoUploader"
import { useSearchParams } from "next/navigation"
import { useEffect, useRef, useState } from "react"

type ProcessingState = "upload" | "processing" | "results"

export default function Home() {
    const [state, setState] = useState<ProcessingState>("upload")
    const [taskId, setTaskId] = useState<string>("")
    const [filename, setFilename] = useState<string>("")
    const [temperatureType, setTemperatureType] = useState<string>("room")
    const [error, setError] = useState<string | null>(null)
    const resultsRef = useRef<HTMLDivElement>(null)
    const searchParams = useSearchParams()

    useEffect(() => {
        const urlTaskId = searchParams.get("taskId")
        if (urlTaskId) {
            setTaskId(urlTaskId)
            setState("results")
        }
    }, [searchParams])

    const handleUploadSuccess = (newTaskId: string, newFilename: string, newTempType: string) => {
        setTaskId(newTaskId)
        setFilename(newFilename)
        setTemperatureType(newTempType)
        setState("processing")
        setError(null)
    }

    const handleUploadError = (errorMessage: string) => {
        setError(errorMessage)
    }

    const handleProcessingComplete = () => {
        setState("results")
        setTimeout(() => {
            resultsRef.current?.scrollIntoView({ behavior: "smooth" })
        }, 100)
    }

    const handleProcessingError = (errorMessage: string) => {
        setError(errorMessage)
        setState("upload")
    }

    const handleNewUpload = () => {
        setState("upload")
        setTaskId("")
        setFilename("")
        setTemperatureType("room")
        setError(null)
        window.scrollTo({ top: 0, behavior: "smooth" })
    }

    return (
        <div className="min-h-screen">
            {/* hero section */}
            <div className="bg-gradient-to-r from-[var(--primary-color)] to-[var(--primary-color-dark)] text-white py-16">
                <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
                    <h1 className="text-4xl font-bold mb-4">
                        AI-Powered Aitbag Timing Detection System
                    </h1>
                    <p className="text-xl opacity-90 max-w-3xl mx-auto">
                        Advanced computer vision technology for precise airbag deployment analysis with temperature specific detection capabilities
                    </p>
                </div>
            </div>

            {/* main content */}
            <div className="max-w-5xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
                {/* error display */}
                {error && (
                    <div className="mb-8 p-4 bg-red-50 border border-red-200 rounded-lg">
                        <p className="text-red-800">{error}</p>
                    </div>
                )}

                {/* Upload section */}
                {state === "upload" && (
                    <div className="space-y-8">
                        <VideoUploader 
                            onUploadSuccess={handleUploadSuccess}
                            onUploadError={handleUploadError}
                        />

                        {/* Features section */}
                        
                    </div>
                )}

                {/* Processing section */}
                {state === "processing" && (
                    <ProcessingProgress 
                        taskId={taskId}
                        filename={filename}
                        temperatureType={temperatureType}
                        onComplete={handleProcessingComplete}
                        onError={handleProcessingError}
                    />
                )}

                {/* results section */}
                {state === "results" && (
                    <div ref={resultsRef}>
                        <ResultsDisplay
                            taskId={taskId}
                            filename={filename}
                            temperatureType={temperatureType}
                            onNewUpload={handleNewUpload}
                        />
                    </div>
                )}
            </div>

            {/* Technical Secifications */}
            {state === "upload" && (
                <div className="bg-gray-50 py-16">
                    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
                        <div className="text-center mb-12">
                            <h2 className="text-3xl font-bold text-gray-900 mb-4">
                                Technical Specifications
                            </h2>
                            <p className="text-lg text-gray-600">
                                Our system uses cutting-edge AI models for comprehensive analysis
                            </p>
                            </div>

                        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-8">
                            <div className="bg-white rounded-lg p-6 shadow-md">
                                <h3 className="font-semibold text-gray-900 mb-2">YOLO Object Detection</h3>
                                <p className="text-sm text-gray-600">
                                Real-time object detection for airbag identification
                                </p>
                            </div>
                            <div className="bg-white rounded-lg p-6 shadow-md">
                                <h3 className="font-semibold text-gray-900 mb-2">SAM Segmentation</h3>
                                <p className="text-sm text-gray-600">
                                Precise mask generation for deployment analysis
                                </p>
                            </div>
                            <div className="bg-white rounded-lg p-6 shadow-md">
                                <h3 className="font-semibold text-gray-900 mb-2">OCR Recognition</h3>
                                <p className="text-sm text-gray-600">
                                Timestamp extraction from video frames
                                </p>
                            </div>
                            <div className="bg-white rounded-lg p-6 shadow-md">
                                <h3 className="font-semibold text-gray-900 mb-2">Multi-Parameter Analysis</h3>
                                <p className="text-sm text-gray-600">
                                Comprehensive detection with multiple validation layers
                                </p>
                            </div>
                        </div>
                    </div>
                </div>
            )}
        </div>
    )
}