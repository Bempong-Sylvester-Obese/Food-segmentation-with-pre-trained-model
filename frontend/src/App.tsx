import { useEffect, useRef, useState } from 'react'
import './App.css'
import { getConfig, getModelStatus, segmentImage } from './api'
import type { AppConfig, ModelStatus, SegmentResponse } from './api'
import { ResultsViewer } from './components/ResultsViewer'
import { StatusCard } from './components/StatusCard'
import { UploadPanel } from './components/UploadPanel'

const FALLBACK_CONFIG: AppConfig = {
  allowed_extensions: ['png', 'jpg', 'jpeg', 'gif', 'bmp'],
  max_image_bytes: 10 * 1024 * 1024,
  max_image_dim: 2048,
  max_image_pixels: 2048 * 2048,
  max_prompt_chars: 200,
  rate_limit: '5/minute',
  api_key_required: false,
}

const PROGRESS_STEPS = ['Upload', 'Detect', 'Segment', 'Render']

function formatError(result: SegmentResponse, status: number, retryAfter: string | null) {
  if (!('error' in result)) return ''
  if (status === 429 && retryAfter) {
    return `${result.error} Retry after ${retryAfter} seconds.`
  }
  if (status === 503) {
    return `${result.error} The model service may still be warming up.`
  }
  return result.error
}

function readImageDimensions(file: File): Promise<string> {
  return new Promise((resolve) => {
    const image = new Image()
    const url = URL.createObjectURL(file)
    image.onload = () => {
      URL.revokeObjectURL(url)
      resolve(`${image.naturalWidth} x ${image.naturalHeight}`)
    }
    image.onerror = () => {
      URL.revokeObjectURL(url)
      resolve('')
    }
    image.src = url
  })
}

function App() {
  const [config, setConfig] = useState<AppConfig | null>(null)
  const [status, setStatus] = useState<ModelStatus | null>(null)
  const [statusLoading, setStatusLoading] = useState(true)
  const [file, setFile] = useState<File | null>(null)
  const [prompt, setPrompt] = useState('jollof rice')
  const [previewUrl, setPreviewUrl] = useState('')
  const [dimensions, setDimensions] = useState('')
  const [error, setError] = useState('')
  const [result, setResult] = useState<SegmentResponse | null>(null)
  const [requestId, setRequestId] = useState('')
  const [isSubmitting, setIsSubmitting] = useState(false)
  const [stepIndex, setStepIndex] = useState(0)
  const abortController = useRef<AbortController | null>(null)

  useEffect(() => {
    let mounted = true

    async function loadConfig() {
      try {
        const loadedConfig = await getConfig()
        if (mounted) setConfig(loadedConfig)
      } catch {
        if (mounted) setConfig(FALLBACK_CONFIG)
      }
    }

    loadConfig()
    return () => {
      mounted = false
    }
  }, [])

  useEffect(() => {
    let mounted = true

    async function refreshStatus() {
      try {
        setStatusLoading(true)
        const loadedStatus = await getModelStatus()
        if (mounted) setStatus(loadedStatus)
      } catch {
        if (mounted) setStatus(null)
      } finally {
        if (mounted) setStatusLoading(false)
      }
    }

    refreshStatus()
    const timer = window.setInterval(refreshStatus, 30000)
    return () => {
      mounted = false
      window.clearInterval(timer)
    }
  }, [])

  useEffect(() => {
    if (!isSubmitting) return
    const timer = window.setInterval(() => {
      setStepIndex((current) => Math.min(current + 1, PROGRESS_STEPS.length - 1))
    }, 1200)
    return () => window.clearInterval(timer)
  }, [isSubmitting])

  async function handleFileSelect(selectedFile: File | null) {
    setError('')
    setResult(null)
    setDimensions('')
    if (previewUrl) URL.revokeObjectURL(previewUrl)

    if (!selectedFile) {
      setFile(null)
      setPreviewUrl('')
      return
    }

    const activeConfig = config ?? FALLBACK_CONFIG
    const extension = selectedFile.name.split('.').pop()?.toLowerCase() ?? ''
    if (!activeConfig.allowed_extensions.includes(extension)) {
      setError(`Upload a supported image type: ${activeConfig.allowed_extensions.join(', ')}.`)
      return
    }
    if (selectedFile.size > activeConfig.max_image_bytes) {
      setError(`Image must be smaller than ${Math.round(activeConfig.max_image_bytes / 1024 / 1024)} MB.`)
      return
    }

    setFile(selectedFile)
    setPreviewUrl(URL.createObjectURL(selectedFile))
    setDimensions(await readImageDimensions(selectedFile))
  }

  async function handleSubmit() {
    const activeConfig = config ?? FALLBACK_CONFIG
    const trimmedPrompt = prompt.trim()
    setError('')

    if (!file) {
      setError('Choose an image before running segmentation.')
      return
    }
    if (!trimmedPrompt) {
      setError('Enter a food prompt before submitting.')
      return
    }
    if (trimmedPrompt.length > activeConfig.max_prompt_chars) {
      setError(`Prompt must be ${activeConfig.max_prompt_chars} characters or fewer.`)
      return
    }

    const controller = new AbortController()
    abortController.current = controller
    setIsSubmitting(true)
    setStepIndex(0)
    setResult(null)
    setRequestId('')

    try {
      const response = await segmentImage(file, trimmedPrompt, controller.signal)
      setResult(response.data)
      setRequestId(response.requestId)
      setError(formatError(response.data, response.status, response.retryAfter))
      setStepIndex(PROGRESS_STEPS.length - 1)
    } catch (submitError) {
      if (submitError instanceof DOMException && submitError.name === 'AbortError') {
        setError('Segmentation request cancelled.')
      } else {
        setError('Unable to reach the segmentation API. Confirm Flask is running on port 5001.')
      }
    } finally {
      setIsSubmitting(false)
      abortController.current = null
    }
  }

  function handleCancel() {
    abortController.current?.abort()
  }

  return (
    <main>
      <nav className="top-nav" aria-label="Primary">
        <a className="brand" href="/">
          FoodVision
        </a>
        <div className="nav-meta">
          <span>{config?.rate_limit ?? FALLBACK_CONFIG.rate_limit} rate limit</span>
          <span>{config?.api_key_required ? 'API key protected' : 'Local mode'}</span>
        </div>
      </nav>

      <header className="hero-panel">
        <div>
          <span className="eyebrow">Prompt-guided food segmentation</span>
          <h1>Detect, isolate, and explain food items in seconds.</h1>
          <p>
            A production-ready interface for GroundingDINO and MobileSAM with request IDs,
            readiness checks, and client-side upload safeguards.
          </p>
        </div>
        <StatusCard status={status} loading={statusLoading} />
      </header>

      <section className="progress-strip" aria-live="polite" aria-label="Segmentation progress">
        {PROGRESS_STEPS.map((step, index) => (
          <span key={step} className={isSubmitting && index <= stepIndex ? 'active' : ''}>
            {step}
          </span>
        ))}
      </section>

      <section className="workspace-grid">
        <UploadPanel
          config={config}
          file={file}
          prompt={prompt}
          previewUrl={previewUrl}
          dimensions={dimensions}
          error={error}
          isSubmitting={isSubmitting}
          onFileSelect={handleFileSelect}
          onPromptChange={setPrompt}
          onSubmit={handleSubmit}
          onCancel={handleCancel}
        />
        <ResultsViewer result={result} requestId={requestId} />
      </section>
    </main>
  )
}

export default App
