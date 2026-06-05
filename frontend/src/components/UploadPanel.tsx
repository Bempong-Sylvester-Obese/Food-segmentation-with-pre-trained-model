import type { AppConfig } from '../api'
import type { DragEvent, KeyboardEvent } from 'react'

const EXAMPLE_PROMPTS = ['jollof rice', 'plantain', 'salad', 'meat', 'banku']

type UploadPanelProps = {
  config: AppConfig | null
  file: File | null
  prompt: string
  previewUrl: string
  dimensions: string
  error: string
  isSubmitting: boolean
  onFileSelect: (file: File | null) => void
  onPromptChange: (prompt: string) => void
  onSubmit: () => void
  onCancel: () => void
}

function formatBytes(bytes?: number) {
  if (!bytes) return '10 MB'
  return `${Math.round(bytes / 1024 / 1024)} MB`
}

export function UploadPanel({
  config,
  file,
  prompt,
  previewUrl,
  dimensions,
  error,
  isSubmitting,
  onFileSelect,
  onPromptChange,
  onSubmit,
  onCancel,
}: UploadPanelProps) {
  const accepted = config?.allowed_extensions.map((extension) => `.${extension}`).join(',') ?? '.png,.jpg,.jpeg,.gif,.bmp'

  function handleDrop(event: DragEvent<HTMLLabelElement>) {
    event.preventDefault()
    onFileSelect(event.dataTransfer.files.item(0))
  }

  function handleKeyDown(event: KeyboardEvent<HTMLLabelElement>) {
    if (event.key === 'Enter' || event.key === ' ') {
      event.preventDefault()
      event.currentTarget.click()
    }
  }

  return (
    <section className="workspace-card upload-card" aria-labelledby="upload-title">
      <div className="section-heading">
        <span className="eyebrow">Workspace</span>
        <h2 id="upload-title">Segment food from a prompt</h2>
        <p>Upload a plate image, describe the food item, and review detection metadata with every result.</p>
      </div>

      <label
        className={`drop-zone ${previewUrl ? 'has-preview' : ''}`}
        onDragOver={(event) => event.preventDefault()}
        onDrop={handleDrop}
        onKeyDown={handleKeyDown}
        tabIndex={0}
      >
        <input
          type="file"
          accept={accepted}
          onChange={(event) => onFileSelect(event.target.files?.item(0) ?? null)}
        />
        {previewUrl ? (
          <img src={previewUrl} alt="Selected upload preview" />
        ) : (
          <div>
            <span className="upload-icon" aria-hidden="true">
              +
            </span>
            <strong>Drop an image or browse files</strong>
            <p>PNG, JPG, JPEG, GIF, or BMP. Max {formatBytes(config?.max_image_bytes)}.</p>
          </div>
        )}
      </label>

      {file ? (
        <div className="file-summary" aria-live="polite">
          <strong>{file.name}</strong>
          <span>
            {formatBytes(file.size)}
            {dimensions ? ` | ${dimensions}` : ''}
          </span>
        </div>
      ) : null}

      <div className="prompt-field">
        <label htmlFor="prompt">Food prompt</label>
        <textarea
          id="prompt"
          value={prompt}
          maxLength={config?.max_prompt_chars ?? 200}
          placeholder="Example: jollof rice"
          onChange={(event) => onPromptChange(event.target.value)}
        />
        <span>
          {prompt.length}/{config?.max_prompt_chars ?? 200} characters
        </span>
      </div>

      <div className="chips" aria-label="Prompt examples">
        {EXAMPLE_PROMPTS.map((example) => (
          <button key={example} type="button" onClick={() => onPromptChange(example)}>
            {example}
          </button>
        ))}
      </div>

      <details className="advanced-panel">
        <summary>Advanced controls</summary>
        <p>Detection thresholds are managed by the API today. This panel is reserved for future tuning controls.</p>
      </details>

      {error ? (
        <div className="alert" role="alert">
          {error}
        </div>
      ) : null}

      <div className="actions">
        <button className="primary-button" type="button" disabled={isSubmitting} onClick={onSubmit}>
          {isSubmitting ? 'Segmenting...' : 'Run segmentation'}
        </button>
        {isSubmitting ? (
          <button className="secondary-button" type="button" onClick={onCancel}>
            Cancel
          </button>
        ) : null}
      </div>
    </section>
  )
}
