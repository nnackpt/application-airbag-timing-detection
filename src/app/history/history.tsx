'use client'

import { useState, useEffect } from 'react'
import { Trash2, Download, Eye, Clock, CheckCircle, XCircle, AlertCircle, Video, Filter, Search } from 'lucide-react'
import { api } from '@/lib/api'
import { VideoRecord } from '@/types/api'
import LoadingSpinner from '@/components/LoadingSpinner'

interface ProcessingHistoryProps {
  refreshTrigger?: number
}

export default function ProcessingHistory({ refreshTrigger }: ProcessingHistoryProps) {
  const [videos, setVideos] = useState<VideoRecord[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [selectedStatus, setSelectedStatus] = useState<string>('all')
  const [searchQuery, setSearchQuery] = useState('')
  const [deletingId, setDeletingId] = useState<string | null>(null)

  const statusOptions = [
    { value: 'all', label: 'All Status' },
    { value: 'completed', label: 'Completed' },
    { value: 'processing', label: 'Processing' },
    { value: 'failed', label: 'Failed' },
    { value: 'pending', label: 'Pending' }
  ]

  const loadHistory = async () => {
    try {
      setLoading(true)
      setError(null)
      const data = await api.getVideoHistory(50)
      setVideos(data)
    } catch (error) {
      console.error('Failed to load history:', error)
      setError('Failed to load processing history')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    loadHistory()
  }, [refreshTrigger])

  const handleDelete = async (taskId: string, filename: string) => {
    if (!confirm(`Are you sure you want to delete "${filename}" and all its associated files?`)) {
      return
    }

    try {
      setDeletingId(taskId)
      await api.deleteVideo(taskId)
      await loadHistory() // Refresh the list
    } catch (error) {
      console.error('Failed to delete video:', error)
      alert('Failed to delete video. Please try again.')
    } finally {
      setDeletingId(null)
    }
  }

  const handleDownload = async (taskId: string, filename: string) => {
    try {
      const blob = await api.downloadVideo(taskId)
      const url = window.URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.style.display = 'none'
      a.href = url
      a.download = `processed_${filename}`
      document.body.appendChild(a)
      a.click()
      window.URL.revokeObjectURL(url)
    } catch (error) {
      console.error('Failed to download video:', error)
      alert('Failed to download video. Please try again.')
    }
  }

  const handleView = (taskId: string) => {
    // Navigate to results page with task ID
    window.location.href = `/?taskId=${taskId}`
  }

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'completed':
        return <CheckCircle className="w-5 h-5 text-green-600" />
      case 'processing':
        return <LoadingSpinner size="sm" className="text-[var(--primary-color)]" />
      case 'failed':
        return <XCircle className="w-5 h-5 text-red-600" />
      case 'pending':
        return <Clock className="w-5 h-5 text-yellow-600" />
      default:
        return <AlertCircle className="w-5 h-5 text-gray-400" />
    }
  }

  const getStatusBadge = (status: string) => {
    const baseClasses = "px-2 py-1 rounded-full text-xs font-medium"
    switch (status) {
      case 'completed':
        return `${baseClasses} bg-green-100 text-green-800`
      case 'processing':
        return `${baseClasses} bg-blue-100 text-blue-800`
      case 'failed':
        return `${baseClasses} bg-red-100 text-red-800`
      case 'pending':
        return `${baseClasses} bg-yellow-100 text-yellow-800`
      default:
        return `${baseClasses} bg-gray-100 text-gray-800`
    }
  }

  const formatDate = (dateString: string): string => {
    const date = new Date(dateString)
    return date.toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    })
  }

  const formatDuration = (seconds: number): string => {
    if (seconds < 60) {
      return `${seconds}s`
    }
    const minutes = Math.floor(seconds / 60)
    const remainingSeconds = seconds % 60
    return `${minutes}m ${remainingSeconds}s`
  }

  const filteredVideos = videos.filter(video => {
    const matchesStatus = selectedStatus === 'all' || video.status === selectedStatus
    const matchesSearch = video.original_filename.toLowerCase().includes(searchQuery.toLowerCase()) ||
                         video.task_id.toLowerCase().includes(searchQuery.toLowerCase())
    return matchesStatus && matchesSearch
  })

  if (loading) {
    return (
      <div className="bg-white rounded-xl shadow-lg p-8 text-center">
        <LoadingSpinner size="lg" className="mx-auto mb-4" />
        <p className="text-gray-600">Loading processing history...</p>
      </div>
    )
  }

  if (error) {
    return (
      <div className="bg-white rounded-xl shadow-lg p-8 text-center">
        <XCircle className="w-12 h-12 text-red-500 mx-auto mb-4" />
        <h3 className="text-lg font-semibold text-gray-900 mb-2">Error Loading History</h3>
        <p className="text-gray-600 mb-4">{error}</p>
        <button
          onClick={loadHistory}
          className="bg-[var(--primary-color)] text-white px-6 py-2 rounded-lg font-medium hover:bg-[var(--primary-dark)] transition-colors"
        >
          Retry
        </button>
      </div>
    )
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="bg-white rounded-xl shadow-lg p-8">
        <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between">
          <div>
            <h1 className="text-2xl font-bold text-gray-900 mb-2">Processing History</h1>
            <p className="text-gray-600">View and manage your airbag detection results</p>
          </div>
          <div className="mt-4 sm:mt-0">
            <button
              onClick={loadHistory}
              className="bg-[var(--primary-color)] text-white px-4 py-2 rounded-lg font-medium hover:bg-[var(--primary-dark)] transition-colors cursor-pointer"
            >
              Refresh
            </button>
          </div>
        </div>

        {/* Filters */}
        <div className="mt-6 flex flex-col sm:flex-row gap-4">
          {/* Status Filter */}
          <div className="relative">
            <Filter className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 w-4 h-4" />
            <select
              value={selectedStatus}
              onChange={(e) => setSelectedStatus(e.target.value)}
              className="pl-10 pr-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-[var(--primary-color)] focus:border-[var(--primary-color)] bg-white min-w-[150px] text-black"
            >
              {statusOptions.map(option => (
                <option key={option.value} value={option.value}>
                  {option.label}
                </option>
              ))}
            </select>
          </div>

          {/* Search */}
          <div className="relative flex-1">
            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 w-4 h-4" />
            <input
              type="text"
              placeholder="Search by filename or task ID..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="w-full pl-10 pr-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-[var(--primary-color)] focus:border-[var(--primary-color)] text-black"
            />
          </div>
        </div>

        {/* Results Count */}
        <div className="mt-4">
          <p className="text-sm text-gray-600">
            Showing {filteredVideos.length} of {videos.length} records
          </p>
        </div>
      </div>

      {/* Video List */}
      {filteredVideos.length === 0 ? (
        <div className="bg-white rounded-xl shadow-lg p-12 text-center">
          <Video className="w-16 h-16 text-gray-300 mx-auto mb-4" />
          <h3 className="text-lg font-semibold text-gray-900 mb-2">
            {videos.length === 0 ? 'No Processing History' : 'No Results Found'}
          </h3>
          <p className="text-gray-600 mb-6">
            {videos.length === 0 
              ? 'Upload and process your first video to see results here.'
              : 'Try adjusting your search or filter criteria.'
            }
          </p>
          {videos.length === 0 && (
            <button
              onClick={() => window.location.href = '/'}
              className="bg-[var(--primary-color)] text-white px-6 py-3 rounded-lg font-medium hover:bg-[var(--primary-dark)] transition-colors cursor-pointer"
            >
              Upload Your First Video
            </button>
          )}
        </div>
      ) : (
        <div className="space-y-4">
          {filteredVideos.map((video) => (
            <div key={video.task_id} className="bg-white rounded-xl shadow-lg p-6">
              <div className="flex flex-col lg:flex-row lg:items-center lg:justify-between">
                {/* Video Info */}
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-3 mb-2">
                    <Video className="w-5 h-5 text-gray-400 flex-shrink-0" />
                    <h3 className="text-lg font-semibold text-gray-900 truncate">
                      {video.original_filename}
                    </h3>
                    <span className={getStatusBadge(video.status)}>
                      {video.status.charAt(0).toUpperCase() + video.status.slice(1)}
                    </span>
                  </div>

                  <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 text-sm text-gray-600 mb-4">
                    <div>
                      <span className="font-medium">Task ID:</span>
                      <p className="font-mono text-xs mt-1">{video.task_id}</p>
                    </div>
                    <div>
                      <span className="font-medium">Created:</span>
                      <p className="mt-1">{formatDate(video.created_at)}</p>
                    </div>
                    <div>
                      <span className="font-medium">Temperature:</span>
                      <p className="mt-1 capitalize">{video.temperature_type || 'N/A'}</p>
                    </div>
                    <div>
                      <span className="font-medium">Status:</span>
                      <p className="mt-1">{video.message || 'N/A'}</p>
                    </div>
                  </div>

                  {/* Progress Bar for Processing Status */}
                  {video.status === 'processing' && (
                    <div className="mb-4">
                      <div className="flex items-center gap-2 mb-1">
                        <span className="text-sm text-gray-600">Processing... {video.progress}%</span>
                        <LoadingSpinner size="sm" />
                      </div>
                      <div className="w-full bg-gray-200 rounded-full h-2">
                        <div 
                          className="bg-[var(--primary-color)] h-2 rounded-full transition-all duration-300"
                          style={{ width: `${video.progress}%` }}
                        />
                      </div>
                    </div>
                  )}

                  {/* Error Message */}
                  {video.status === 'failed' && video.message && (
                    <div className="mb-4 p-3 bg-red-50 border border-red-200 rounded-lg">
                      <p className="text-sm text-red-800">{video.message}</p>
                    </div>
                  )}

                  {/* Results Summary */}
                  {video.status === 'completed' && (
                    <div className="mb-4 p-3 bg-green-50 border border-green-200 rounded-lg">
                      <p className="text-sm text-green-800">
                        <span className="font-medium">Processing completed successfully</span>
                        {video.screenshots && video.screenshots.length > 0 && (
                          <span> • {video.screenshots.length} screenshot(s) generated</span>
                        )}
                      </p>
                    </div>
                  )}
                </div>

                {/* Actions */}
                <div className="flex items-center gap-2 mt-4 lg:mt-0 lg:ml-6">
                  {/* Status Icon */}
                  <div className="flex items-center gap-2">
                    {getStatusIcon(video.status)}
                  </div>

                  {/* Action Buttons */}
                  <div className="flex gap-1">
                    {video.status === 'completed' && (
                      <>
                        <button
                          onClick={() => handleView(video.task_id)}
                          className="p-2 text-gray-600 hover:text-[var(--primary-color)] hover:bg-[var(--primary-color)]/10 rounded-lg transition-colors"
                          title="View Results"
                        >
                          <Eye className="w-4 h-4" />
                        </button>
                        <button
                          onClick={() => handleDownload(video.task_id, video.original_filename)}
                          className="p-2 text-gray-600 hover:text-blue-600 hover:bg-blue-50 rounded-lg transition-colors"
                          title="Download"
                        >
                          <Download className="w-4 h-4" />
                        </button>
                      </>
                    )}
                    <button
                      onClick={() => handleDelete(video.task_id, video.original_filename)}
                      disabled={deletingId === video.task_id}
                      className="p-2 text-gray-600 hover:text-red-600 hover:bg-red-50 rounded-lg transition-colors disabled:opacity-50 cursor-pointer"
                      title="Delete"
                    >
                      {deletingId === video.task_id ? (
                        <LoadingSpinner size="sm" />
                      ) : (
                        <Trash2 className="w-4 h-4" />
                      )}
                    </button>
                  </div>
                </div>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Load More Button */}
      {videos.length > 0 && filteredVideos.length === videos.length && videos.length >= 50 && (
        <div className="text-center">
          <button
            onClick={() => {
              // This would typically load more records
              // You might want to implement pagination here
              console.log('Load more videos')
            }}
            className="bg-gray-100 text-gray-700 px-6 py-3 rounded-lg font-medium hover:bg-gray-200 transition-colors"
          >
            Load More
          </button>
        </div>
      )}
    </div>
  )
}