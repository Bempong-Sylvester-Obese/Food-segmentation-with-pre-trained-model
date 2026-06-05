import type { SegmentResponse } from '../api'

type ResultsViewerProps = {
  result: SegmentResponse | null
  requestId: string
}

function imageSrc(image: string) {
  return `data:image/png;base64,${image}`
}

function downloadImage(image: string, filename: string) {
  const link = document.createElement('a')
  link.href = imageSrc(image)
  link.download = filename
  link.click()
}

export function ResultsViewer({ result, requestId }: ResultsViewerProps) {
  if (!result) {
    return (
      <section className="workspace-card empty-state" aria-label="Results">
        <span className="eyebrow">Results</span>
        <h2>Ready for the first segmentation</h2>
        <p>Your before/after viewer, request ID, and detection metadata will appear here.</p>
      </section>
    )
  }

  if ('error' in result) {
    return (
      <section className="workspace-card empty-state" aria-label="Segmentation error">
        <span className="eyebrow">No result</span>
        <h2>{result.code === 'no_detections' ? 'No matching food detected' : 'Segmentation did not complete'}</h2>
        <p>{result.error}</p>
        {requestId ? <code>Request ID: {requestId}</code> : null}
      </section>
    )
  }

  const metadata = result.metadata
  const detections = metadata?.detections

  return (
    <section className="workspace-card results-card" aria-label="Segmentation results">
      <div className="section-heading">
        <span className="eyebrow">Results</span>
        <h2>Before and after</h2>
        <p>Review the segmentation overlay and keep the request ID for debugging.</p>
      </div>

      <div className="image-grid">
        <figure>
          <img src={imageSrc(result.original_image)} alt="Original uploaded food" />
          <figcaption>Original</figcaption>
        </figure>
        <figure>
          <img src={imageSrc(result.result_image)} alt="Food segmentation result overlay" />
          <figcaption>Segmented overlay</figcaption>
        </figure>
      </div>

      <div className="metadata-grid">
        <div>
          <span>Detections</span>
          <strong>{detections?.count ?? 0}</strong>
        </div>
        <div>
          <span>Duration</span>
          <strong>{metadata?.duration_ms ? `${metadata.duration_ms} ms` : 'n/a'}</strong>
        </div>
        <div>
          <span>Image</span>
          <strong>
            {metadata?.image ? `${metadata.image.width} x ${metadata.image.height}` : 'n/a'}
          </strong>
        </div>
        <div>
          <span>Device</span>
          <strong>{metadata?.device ?? 'n/a'}</strong>
        </div>
      </div>

      {detections?.phrases?.length ? (
        <div className="phrase-list">
          <span>Detected phrases</span>
          <p>{detections.phrases.join(', ')}</p>
        </div>
      ) : null}

      <div className="actions wrap-actions">
        <button type="button" className="secondary-button" onClick={() => downloadImage(result.original_image, 'original.png')}>
          Download original
        </button>
        <button type="button" className="secondary-button" onClick={() => downloadImage(result.result_image, 'segmented.png')}>
          Download result
        </button>
        <button
          type="button"
          className="secondary-button"
          onClick={() => navigator.clipboard.writeText(JSON.stringify({ requestId, result }, null, 2))}
        >
          Copy API response
        </button>
      </div>

      {requestId ? <code>Request ID: {requestId}</code> : null}
    </section>
  )
}
