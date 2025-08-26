import { api } from "@/lib/api"
import { DetectionResult, Screenshot } from "@/types/api"
import { useEffect, useState } from "react"
import LoadingSpinner from "./LoadingSpinner"
import { BarChart3, ChevronDown, ChevronUp, Clock, Download, FileText, ImageIcon, Target, Zap } from "lucide-react"

interface ResultsDisplayProps {
    taskId: string
    filename: string
    temperatureType: string
    onNewUpload: () => void
}

export default function ResultsDisplay({
    taskId,
    filename,
    temperatureType,
    onNewUpload
}: ResultsDisplayProps) {
    const [results, setResults] = useState<DetectionResult | null>(null)
    const [screenshots, setScreenshots] = useState<Screenshot[]>([])
    const [loading, setLoading] = useState(true)
    const [error, setError] = useState<string | null>(null)
    const [selectedImage, setSelectedImage] = useState<string | null>(null)
    const [showOCRResults, setShowOCRResults] = useState(false)

    useEffect(() => {
        const loadResults = async () => {
            try {
                setLoading(true)
                const [resultsData, screenshotsData] = await Promise.all([
                    api.getDetectionResults(taskId),
                    api.getScreenshots(taskId)
                ])

                setResults(resultsData)
                setScreenshots(screenshotsData.screenshots)
            } catch (err) {
                console.error("Failed to load results:", err)
                setError("Failed to load detection results")
            } finally {
                setLoading(false)
            }
        }

        loadResults()
    }, [taskId])

    const handleDownloadVideo = async () => {
        try {
            const blob = await api.downloadVideo(taskId)
            const url = window.URL.createObjectURL(blob)
            const a = document.createElement("a")
            a.style.display = "none"
            a.href = url
            a.download = `processed_${filename}`
            document.body.appendChild(a)
            a.click()
            window.URL.revokeObjectURL(url)
        } catch (err) {
            console.error("Failed to download video:", err)
        }
    }

    const handleDownloadScreenshot = async (screenshot: Screenshot) => {
        try {
            const blob = await api.downloadScreenshot(taskId, screenshot.filename)
            const url = window.URL.createObjectURL(blob)
            const a = document.createElement("a")
            a.style.display = "none"
            a.href = url
            a.download = screenshot.filename
            document.body.appendChild(a)
            a.click()
            window.URL.revokeObjectURL(url)
        } catch (err) {
            console.error("Failed to download screenshot:", err)
        }
    }

    const formatTime = (seconds: number): string => {
        const mins = Math.floor(seconds / 60)
        const secs = Math.floor(seconds % 60)
        return `${mins}:${secs.toString().padStart(2, '0')}` 
    }

    if (loading) {
        return (
            <div className="bg-white rounded-xl shadow-lg p-8 text-center">
                <LoadingSpinner size="lg" className="mx-auto mb-4" />
                <p className="text-gray-600">Loading detection results...</p>
            </div>
        )
    }

    if (error || !results) {
        return (
            <div className="bg-white rounded-xl shadow-lg p-8 text-center">
                <div className="text-red-500 mb-4">
                    <Target className="w-12 h-12 mx-auto" />
                </div>
                <h3 className="text-lg font-semibold text-gray-900 mb-2">Failed to Load Results</h3>
                <p className="text-gray-600 mb-4">{error}</p>
                <button
                    onClick={onNewUpload}
                    className="bg-[var(--primary-color)] text-white px-6 py-2 rounded-lg font-medium hover:bg-[var(--primary-color-dark)] transition-colors"
                >
                    Try Another Video
                </button>
            </div>
        )
    }

    const detectedLabels = ["FR1", "FR2", "RE3"]
    const ocrResults = results.ocr_results || {}

    return (
        <div className="space-y-6">
            {/* success header */}
            <div className="bg-white rounded-xl shadow-;g p-8 text-center">
                <div className="w-16 h-16 bg-green-100 rounded-full flex items-center justify-center mx-auto mb-4">
                    <Target className="w-8 h-8 text-green-600" />
                </div>
                <h2 className="text-2xl font-bold text-gray-900 mb-2">Analysis Complete!</h2>
                <p className="text-gray-600 mb-4">
                    Successfully processed <span className="font-medium">{filename}</span>
                    <span className="text-[var(--primary-color)]">({temperatureType} temperature)</span>
                </p>

                <div className="flex flex-wrap justify-center gap-4 mb-6">
                    <div className="bg-blue-50 px-4 py-2 rounded-lg">
                        <div className="flex items-center text-[var(--primary-color)]">
                            <Clock className="w-4 h-4 mr-2" />
                            <span className="text-sm font-medium">
                                Processing Time: {formatTime(results.processing_time)}
                            </span>
                        </div>
                    </div>
                    <div className="bg-green-50 px-4 py-2 rounded-lg">
                        <div className="flex items-center text-green-600">
                            <Target className="w-4 h-4 mr-2" />
                            <span className="text-sm font-medium">
                                Labels Detected: {detectedLabels.length}/3
                            </span>
                        </div>
                    </div>
                </div>

                <button
                    onClick={handleDownloadVideo}
                    className="bg-[var(--primary-color)] text-white px-6 py-3 rounded-lg font-semibold 
                                hover:bg-[var(--primary-color-dark)] transition-colors flex items-center space-x-2 mx-auto"
                >
                    <Download className="w-5 h-5" />
                    <span>Download Processed Video</span>
                </button>
            </div>
            
            {/* Detection Summary */}
            <div className="bg-white rounded-xl shadow-lg p-8">
                <h3 className="text-xl font-bold text-gray-900 mb-6">Detection Summary</h3>

                <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                    <div className="text-center p-4 bg-orange-50 rounded-lg">
                        <Zap className="w-8 h-8 text-orange-600 mx-auto mb-2" />
                        <h4 className="font-semibold text-gray-900 mb-1">Explosion Frame</h4>
                        <p className="text-2xl font-bold text-orange-600">
                            {results.explosion_frame || "N/A"}
                        </p>
                    </div>

                    <div className="text-center p-4 bg-green-50 rounded-lg">
                        <BarChart3 className="w-8 h-8 text-green-600 mx-auto mb-2" />
                        <h4 className="font-semibold text-gray-900 mb-1">Full Deployment</h4>
                        <p className="text-2xl font-bold text-green-600">
                            {results.full_deployment_frame || "N/A"}
                        </p>
                    </div>

                    <div className="text-center p-4 bg-blue-50 rounded-lg">
                        <Target className="w-8 h-8 text-[var(--primary-color)] mx-auto mb-2" />
                        <h4 className="font-semibold text-gray-900 mb-1">Detection Rate</h4>
                        <p className="text-2xl font-bold text-[var(--primary-color)]">
                            {Math.round((detectedLabels.length / 3) * 100)}%
                        </p>
                    </div>
                </div>
            </div>

            {/* Screenshot gallery */}
            <div className="bg-white rounded-xl shadow-lg p-8">
                <h3 className="text-xl font-bold text-gray-900 mb-6">Captured Screenshots</h3>

                {screenshots.length > 0 ? (
                    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                        {screenshots.map((screenshot, index) => (
                            <div key={index} className="bg-gray-50 rounded-lg p-4">
                                <div 
                                    className="aspect-video bg-gray-200 rounded-lg mb-4 overflow-hidden cursor-pointer"
                                    onClick={() => setSelectedImage(api.getScreenshotUrl(taskId, screenshot.filename))}
                                >
                                    <img  
                                        src={api.getScreenshotUrl(taskId, screenshot.filename)}
                                        alt={screenshot.filename}
                                        className="w-full h-full object-cover hover:scale-105 transition-transform duration-200"
                                    />
                                </div>
                                <div className="flex items-center justify-between">
                                    <div>
                                        <h4 className="font-medium text-gray-900 text-sm truncate">
                                            {screenshot.filename}
                                        </h4>
                                        <p className="text-xs text-gray-500 mt-1">
                                            {screenshot.filename.includes('FR1') && 'Front Left'}
                                            {screenshot.filename.includes('FR2') && 'Front Right'}
                                            {screenshot.filename.includes('RE3') && 'Rear'}
                                            {screenshot.filename.includes('Explosion') && 'Explosion Detection'}
                                            {screenshot.filename.includes('Full_Deployment') && 'Full Deployment'}
                                        </p>
                                    </div>
                                    <button
                                        onClick={() => handleDownloadScreenshot(screenshot)}
                                        className="p-2 text-gray-400 hover:text-[var(--primary-color)] transition-colors"
                                        title="Download"
                                    >
                                        <Download className="w-4 h-4" />
                                    </button>
                                </div>
                            </div>
                        ))}
                    </div>
                ) : (
                    <div className="text-center py-8">
                        <ImageIcon className="w-12 h-12 text-gray-400 mx-auto mb-4" />
                        <p className="text-gray-500">No screenshots available</p>
                    </div>
                )}
            </div>

            {/* Ocr */}
            {Object.keys(ocrResults).length > 0 && (
                <div className="bg-white rounded-xl shadow-lg p-8">
                    <button
                        onClick={() => setShowOCRResults(!showOCRResults)}
                        className="flex items-center justify-between w-full text-left"
                    >
                        <h3 className="text-xl font-bold text-gray-900 flex items-center">
                            <FileText className="w-6 h-6 mr-2" />
                            OCR Results
                        </h3>
                        {showOCRResults ? (
                            <ChevronUp className="w-5 h-5 text-gray-500" />
                        ) : (
                            <ChevronDown className="w-5 h-5 text-gray-500" />
                        )}
                    </button>

                    {showOCRResults && (
                        <div className="mt-6 space-y-4">
                            {Object.entries(ocrResults).map(([filename, result]) => (
                                <div key={filename} className="border rounded-lg p-4">
                                    <h4 className="font-medium text-gray-900 mb-2">{filename}</h4>
                                    <p className="text-sm text-gray-600 font-mono bg-gray-50 p-2 rounded">
                                        {result}
                                    </p>
                                </div>
                            ))}
                        </div>
                    )}
                </div>
            )}

            {/* Image modal */}
            {selectedImage && (
                <div
                    className="fixed inset-0 bg-black bg-opacity-75 flex items-center justify-center z-50 p-4"
                    onClick={() => setSelectedImage(null)}
                >
                    <div className="relative max-w-4xl max-h-full">
                        <img 
                            src={selectedImage}  
                            alt="Screenshot"
                            className="max-w-full max-h-full object-contain" 
                        />
                        <button
                            onClick={() => setSelectedImage(null)}
                            className="absolute top-4 right-4 text-white hover:text-gray-300"
                        >
                            <span className="text-2xl">&times;</span>
                        </button>
                    </div>
                </div>
            )}

            {/* action button */}
            <div className="text-center">
                <button
                    onClick={onNewUpload}
                    className="bg-[var(--primary-color)] text-white px-8 py-3 rounded-lg font-semibold hover:bg-[var(--primary-color-dark)] transition-colors"
                >
                    Analyze Another Video
                </button>
            </div>
        </div>
    )
}