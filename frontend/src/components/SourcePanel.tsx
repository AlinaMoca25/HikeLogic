import type { SourceItem } from "../types";

interface SourcePanelProps {
  sources: SourceItem[];
}

function metaTag(value: unknown): string | null {
  if (value == null || value === "" || value === "unknown") return null;
  return String(value);
}

export default function SourcePanel({ sources }: SourcePanelProps) {
  return (
    <aside className="sources-column">
      <div className="sources-header">
        <h2>Surse din baza de trasee</h2>
        <p>
          {sources.length > 0
            ? `${sources.length} fragmente relevante (OpenStreetMap)`
            : "După o întrebare, sursele recuperate apar aici."}
        </p>
      </div>

      <div className="sources-list">
        {sources.length === 0 ? (
          <div className="sources-empty">
            Răspunsurile HikeLogic sunt ancorate în trasee și POI-uri din
            România. Întreabă despre cabane, dificultate, izvoare sau zone
            montane.
          </div>
        ) : (
          sources.map((source) => {
            const name = metaTag(source.metadata.name) ?? "Fără nume";
            const difficulty = metaTag(source.metadata.difficulty);
            const region = metaTag(source.metadata.region);
            const entityType = metaTag(source.metadata.entity_type);
            const osmUrl = metaTag(source.metadata.osm_url);

            return (
              <details key={source.index} className="source-card" open>
                <summary>
                  <span className="source-index">[{source.index}]</span>
                  <span className="source-name">{name}</span>
                  <span className="source-score">
                    relevanță {source.score.toFixed(3)}
                  </span>
                  <div className="source-tags">
                    {entityType && <span className="tag">{entityType}</span>}
                    {difficulty && <span className="tag">{difficulty}</span>}
                    {region && <span className="tag">{region}</span>}
                  </div>
                </summary>
                <div className="source-body">
                  {source.text.slice(0, 600)}
                  {source.text.length > 600 ? "…" : ""}
                  {osmUrl && (
                    <a
                      className="source-link"
                      href={osmUrl}
                      target="_blank"
                      rel="noreferrer"
                    >
                      Vezi pe OpenStreetMap →
                    </a>
                  )}
                </div>
              </details>
            );
          })
        )}
      </div>
    </aside>
  );
}
