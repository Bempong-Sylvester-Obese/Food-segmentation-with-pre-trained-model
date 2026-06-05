import type { ModelStatus } from '../api'

type StatusCardProps = {
  status: ModelStatus | null
  loading: boolean
}

export function StatusCard({ status, loading }: StatusCardProps) {
  const ready = Boolean(
    status?.models_loaded.grounding_dino && status.models_loaded.sam_predictor,
  )

  return (
    <section className="status-card" aria-label="Model readiness">
      <div>
        <span className={`status-dot ${ready ? 'ready' : 'not-ready'}`} aria-hidden="true" />
        <span className="eyebrow">Model service</span>
      </div>
      <strong>{loading ? 'Checking...' : ready ? 'Ready' : status?.status ?? 'Unknown'}</strong>
      <p>
        {ready
          ? `GroundingDINO and MobileSAM are loaded on ${status?.device ?? 'the active device'}.`
          : 'Uploads are accepted when the model service reports ready.'}
      </p>
    </section>
  )
}
