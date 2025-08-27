import { VideoUploadResponse, ProcessingStatus, VideoRecord, DetectionResult, TemperatureOption, Screenshot } from '@/types/api';

const API_BASE_URL = '/fastapi';

export class ApiError extends Error {
  constructor(public status: number, message: string) {
    super(message);
    this.name = 'ApiError';
  }
}

async function handleResponse<T>(response: Response): Promise<T> {
  if (!response.ok) {
    const errorText = await response.text();
    throw new ApiError(response.status, errorText || response.statusText);
  }
  return response.json();
}

export const api = {
  async uploadVideo(file: File, temperatureType: string): Promise<VideoUploadResponse> {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('temperature_type', temperatureType);

    const response = await fetch(`${API_BASE_URL}/upload`, {
      method: 'POST',
      body: formData,
    });

    return handleResponse<VideoUploadResponse>(response);
  },

  async getProcessingStatus(taskId: string): Promise<ProcessingStatus> {
    const response = await fetch(`${API_BASE_URL}/status/${taskId}`);
    return handleResponse<ProcessingStatus>(response);
  },

  async getVideoHistory(limit: number = 50): Promise<VideoRecord[]> {
    const response = await fetch(`${API_BASE_URL}/videos?limit=${limit}`);
    return handleResponse<VideoRecord[]>(response);
  },

  async getDetectionResults(taskId: string): Promise<DetectionResult> {
    const response = await fetch(`${API_BASE_URL}/results/${taskId}`);
    return handleResponse<DetectionResult>(response);
  },

  async getScreenshots(taskId: string): Promise<{ screenshots: Screenshot[] }> {
    const response = await fetch(`${API_BASE_URL}/screenshots/${taskId}`);
    return handleResponse<{ screenshots: Screenshot[] }>(response);
  },

  async getTemperatureOptions(): Promise<{ options: TemperatureOption[] }> {
    const response = await fetch(`${API_BASE_URL}/temperature-options`);
    return handleResponse<{ options: TemperatureOption[] }>(response);
  },

  async downloadVideo(taskId: string): Promise<Blob> {
    const response = await fetch(`${API_BASE_URL}/video/${taskId}`);
    if (!response.ok) {
      throw new ApiError(response.status, 'Failed to download video');
    }
    return response.blob();
  },

  async downloadScreenshot(taskId: string, filename: string): Promise<Blob> {
    const response = await fetch(`${API_BASE_URL}/screenshot/${taskId}/${filename}`);
    if (!response.ok) {
      throw new ApiError(response.status, 'Failed to download screenshot');
    }
    return response.blob();
  },

  async deleteVideo(taskId: string): Promise<{ message: string }> {
    const response = await fetch(`${API_BASE_URL}/video/${taskId}`, {
      method: 'DELETE',
    });
    return handleResponse<{ message: string }>(response);
  },

  getVideoUrl(taskId: string): string {
    return `${API_BASE_URL}/video/${taskId}`;
  },

  getScreenshotUrl(taskId: string, filename: string): string {
    return `${API_BASE_URL}/screenshot/${taskId}/${filename}`;
  }
};