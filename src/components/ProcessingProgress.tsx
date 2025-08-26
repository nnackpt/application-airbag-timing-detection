import { api } from "@/lib/api"
import { ProcessingStatus } from "@/types/api"
import { AlertCircle, BarChart3, CheckCircle, Clock, Search, Target, Video, Zap } from "lucide-react"
import { useEffect, useState } from "react"
import LoadingSpinner from "./LoadingSpinner"

interface ProcessingProgressProps {
    taskId: string
    filename: string
    temperatureType: string
    onComplete: () => void
    onError: (error: string) => void
}

export default function ProcessingProgress({
    taskId,
    filename,
    temperatureType,
    onComplete,
    onError
}: ProcessingProgressProps) {
    const [status, setStatus] = useState<ProcessingStatus | null>(null)
    const [currentStep, setCurrentStep] = useState(0)

    const steps = [
        {
            id: 1,
            name: "Video Upload",
            description: "Video uploaded successfully",
            icon: Video,
            completed: true
        },
        {
            id: 2,
            name: "Circle Detection",
            description: "Detecting airbag deployment circles with enhanced consensus algorithm",
            icon: Target,
            completed: false
        },
        {
            id: 3,
            name: "Label Detection",
            description: "Identifying FR1, FR2, RE3, labels and matching to circles",
            icon: Search,
            completed: false
        },
        {
            id: 4,
            name: "Object Analysis",
            description: "Analyzing airbag deployment objects and explosion detection",
            icon: Zap,
            completed: false
        },
        {
            id: 5,
            name: "Full Deployment",
            description: `Analyzing full deployment for ${temperatureType} temperature`,
            icon: BarChart3,
            completed: false
        }
    ]

    useEffect(() => {
        if (!taskId) return

        const pollStatus = async () => {
            try {
                const statusData = await api.getProcessingStatus(taskId)
                setStatus(statusData)

                // Update current step based on progress and message
                if (statusData.progress <= 10) {
                    setCurrentStep(1)
                } else if (statusData.progress <= 30) {
                    setCurrentStep(2)
                } else if (statusData.progress <= 50) {
                    setCurrentStep(3)
                } else if (statusData.progress <= 80) {
                    setCurrentStep(4)
                } else if (statusData.progress <= 100) {
                    setCurrentStep(5)
                }

                if (statusData.status === "completed") {
                    onComplete()
                    return
                } else if (statusData.status === "failed") {
                    onError(statusData.message)
                    return
                }
            } catch (err) {
                console.error("Failed to fetch status:", err)
                onError("Failed to fetch processing status")
                return
            }
        }

        pollStatus()

        const interval = setInterval(pollStatus, 2000)

        return () => clearInterval(interval)
    }, [taskId, onComplete, onError])

    if (!status) {
        return (
            <div className="bg-white rounded-xl shadow-lg p-8 text-center">
                <LoadingSpinner size="lg" className="mx-auto mb-4" />
                <p className="text-gray-600">Loading processing status...</p>
            </div>
        )
    }

    const getStepStatus = (stepId: number) => {
        if (stepId < currentStep) return "completed"
        if (stepId === currentStep) return "current"
        return "pending"
    }

    return (
        <div className="bg-white rounded-xl shadow-lg p-8">
            <div className="text-center mb-8">
                <h2 className="text-2xl font-bold text-gray-900 mb-2">Processing Your Video</h2>
                <p className="text-gray-600">{filename}</p>
                <p className="text-sm text-[var(--primary-color)] font-medium mt-1">
                    {temperatureType.charAt(0).toUpperCase() + temperatureType.slice(1)} Temperature Analysis
                </p>
            </div>

            {/* Progress bar */}
            <div className="mb-8">
                <div className="flex justify-between items-center mb-2">
                    <span className="text-sm font-medium text-gray-700">Overall Progress</span>
                    <span className="text-sm font-medium text-[var(--primary-color)]">{status.progress}%</span>
                </div>
                <div className="w-full bg-gray-200 rounded-full h-3">
                    <div
                        className="bg-[var(--primary-color)] h-3 rounded-full transition-all duration-500 ease-out"
                        style={{ width: `${status.progress}%` }}
                    ></div>
                </div>
            </div>

            {/* current status message */}
            <div className="bg-blue-50 border border-blue-200 rounded-lg p-4 mb-8">
                <div className="flex items-center">
                    <div className="flex-shrink-0">
                        {status.status === "processing" ? (
                            <LoadingSpinner size="sm" className="text-[var(--primary-color)]" />
                        ) : status.status === "completed" ? (
                            <CheckCircle className="w-5 h-5 text-green-500" />
                        ) : status.status === "failed" ? (
                            <AlertCircle className="w-5 h-5 text-red-500" />
                        ) : (
                            <Clock className="w-5 h-5 text-yellow-600" />
                        )}
                    </div>
                    <div className="ml-3">
                        <p className="text-sm font-medium text-gray-900">
                            {status.message}
                        </p>
                    </div>
                </div>
            </div>

            {/* step progress */}
            <div className="space-y-4">
                {steps.map((step) => {
                    const stepStatus = getStepStatus(step.id)
                    const Icon = step.icon

                    return (
                        <div
                            key={step.id}
                            className={`flex items-start p-4 rounded-lg border transition-all duration-300 ${
                                stepStatus === "completed"
                                    ? "bg-green-50 border-green-200"
                                    : stepStatus === "current"
                                    ? "bg-blue-50 border-blue-200 shadow-md"
                                    : "bg-gray-50 border-gray-200"
                            }`}
                        >
                            <div className="flex-shrink-0 mr-4">
                                <div className={`w-10 h-10 rounded-full flex items-center justify-center ${
                                    stepStatus === "completed"
                                        ? "bg-green-500"
                                        : stepStatus === "current"
                                        ? "bg-[var(--primary-color)]"
                                        : "bg-gray-300"
                                    }`}
                                >
                                    {stepStatus === "completed" ? (
                                        <CheckCircle className="w-5 h-5 text-white" />
                                    ) : stepStatus === "current" ? (
                                        <LoadingSpinner size="sm" className="text-white" />
                                    ) : (
                                        <Icon className="w-5 h-5 text-white" />
                                    )}
                                </div>
                            </div>
                            <div className="flex-1 min-w-0">
                                <h3 className={`text-sm font-semibold ${
                                    stepStatus === "completed"
                                        ? "text-green-900" 
                                        : stepStatus === "current"
                                        ? "text-blue-900"
                                        : "text-gray-500"
                                    }`}
                                >
                                    {step.name}
                                    {stepStatus === "completed" && (
                                        <span className="ml-2 inline-flex items-center px-2 py-0.5 rounded text-xs font-medium bg-green-100 text-green-800">
                                            Completed
                                        </span>
                                    )}
                                    {stepStatus === "current" && (
                                        <span className="ml-2 inline-flex items-center px-2 py-0.5 rounded text-xs font-medium bg-blue-100 text-blue-800">
                                            In Progress
                                        </span>
                                    )}
                                </h3>
                                <p className={`text-sm mt-1 ${
                                    stepStatus === "completed"
                                        ? "text-green-700"
                                        : stepStatus === "current"
                                        ? "text-blue-700"
                                        : "text-gray-500"
                                    }`}
                                >
                                    {step.description}
                                </p>
                            </div>
                        </div>
                    )
                })}
            </div>

            {/* processing details */}
            <div className="mt-8 p-4 bg-gray-50 rounded-lg">
                <h4 className="text-sm font-semibold text-gray-900 mb-2">Detection Features</h4>
                <ul className="text-xs text-gray-600 space-y-1">
                    <li>• Center Consensus with Clustering Algorithm</li>
                    <li>• Hungarian Algorithm for Optimal Label-Circle Matching</li>
                    <li>• Distance-based Gating System</li>
                    <li>• Multi-parameter Circle Detection</li>
                    <li>• Temperature-specific Frame Analysis</li>
                </ul>
            </div>
        </div>
    )
} 