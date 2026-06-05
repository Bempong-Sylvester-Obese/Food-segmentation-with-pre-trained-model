export type AppConfig = {
  allowed_extensions: string[]
  max_image_bytes: number
  max_image_dim: number
  max_image_pixels: number
  max_prompt_chars: number
  rate_limit: string
  api_key_required: boolean
}

export type ModelStatus = {
  status: string
  models_loaded: {
    grounding_dino: boolean
    sam_predictor: boolean
  }
  device: string
  torch_available: boolean
}

export type SegmentMetadata = {
  duration_ms?: number
  image?: {
    width: number
    height: number
  }
  detections?: {
    count: number
    boxes: number[][]
    confidence: number[]
    phrases: string[]
  }
  device?: string
}

export type SegmentSuccess = {
  success: true
  original_image: string
  result_image: string
  metadata?: SegmentMetadata
}

export type SegmentFailure = {
  success: false
  error: string
  code?: string
  metadata?: SegmentMetadata
}

export type SegmentResponse = SegmentSuccess | SegmentFailure

export async function getConfig(): Promise<AppConfig> {
  const response = await fetch('/api/v1/config')
  if (!response.ok) {
    throw new Error('Unable to load application configuration.')
  }
  return response.json()
}

export async function getModelStatus(): Promise<ModelStatus> {
  const response = await fetch('/api/v1/models/status')
  if (!response.ok) {
    throw new Error('Unable to load model status.')
  }
  return response.json()
}

export async function segmentImage(
  image: File,
  prompt: string,
  signal: AbortSignal,
): Promise<{ data: SegmentResponse; requestId: string; status: number; retryAfter: string | null }> {
  const formData = new FormData()
  formData.append('image_file', image)
  formData.append('prompt', prompt)

  const requestId = crypto.randomUUID()
  const response = await fetch('/api/v1/segment', {
    method: 'POST',
    body: formData,
    headers: {
      'X-Request-ID': requestId,
    },
    signal,
  })

  let data: SegmentResponse
  try {
    data = (await response.json()) as SegmentResponse
  } catch {
    data = {
      success: false,
      error: response.ok ? 'The API returned an unreadable response.' : 'The server returned a non-JSON error.',
      code: 'invalid_response',
    }
  }

  return {
    data,
    requestId: response.headers.get('X-Request-ID') || requestId,
    status: response.status,
    retryAfter: response.headers.get('Retry-After'),
  }
}
