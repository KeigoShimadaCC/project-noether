"use client";

import { useRouter } from "next/navigation";
import Link from "next/link";
import { useEffect, useState } from "react";
import Latex from "@/components/Latex";
import { api, ApiError } from "@/lib/api";

const DEFAULT_MEASURE = "d^4x \\sqrt{-g}";

export default function HomePage() {
  const router = useRouter();
  const [lagrangian, setLagrangian] = useState("");
  const [measure, setMeasure] = useState(DEFAULT_MEASURE);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [sessions, setSessions] = useState<string[]>([]);

  useEffect(() => {
    api
      .listSessions()
      .then((body) => setSessions(body.sessions))
      .catch(() => setSessions([]));
  }, []);

  async function submit(event: React.FormEvent) {
    event.preventDefault();
    if (!lagrangian.trim()) return;
    setBusy(true);
    setError(null);
    try {
      const session = await api.createSession(lagrangian, measure || undefined);
      router.push(`/sessions/${session.session_id}`);
    } catch (err) {
      setError(err instanceof ApiError ? String(err.message) : "request failed; is the API server running? (noether serve)");
      setBusy(false);
    }
  }

  return (
    <>
      <div className="card">
        <h2>New session</h2>
        <p className="note">
          Paste a scalar Lagrangian density in LaTeX. Noether parses it, asks
          the clarifying questions it cannot decide for you, and only then
          plans the derivation.
        </p>
        <form onSubmit={submit}>
          <textarea
            value={lagrangian}
            onChange={(event) => setLagrangian(event.target.value)}
            placeholder={"F(\\phi) R - \\tfrac12 \\nabla_\\mu\\phi \\nabla^\\mu\\phi - V(\\phi)"}
            aria-label="Lagrangian density"
          />
          <div className="freeform">
            <input
              type="text"
              value={measure}
              onChange={(event) => setMeasure(event.target.value)}
              aria-label="Measure"
            />
            <button type="submit" disabled={busy || !lagrangian.trim()}>
              {busy ? "Ingesting..." : "Ingest"}
            </button>
          </div>
        </form>
        <div className="preview">
          <span className="note">Live preview</span>
          <Latex
            tex={`S \\;=\\; \\int ${measure || "d^4x"} \\, \\left( ${
              lagrangian.trim() || "\\mathcal{L}"
            } \\right)`}
            block
          />
        </div>
        {error && <div className="error-box">{error}</div>}
      </div>

      <div className="card">
        <h2>Stored sessions</h2>
        {sessions.length === 0 ? (
          <p className="note">None yet. Sessions are shared with the CLI and MCP frontends.</p>
        ) : (
          <ul className="session-list">
            {sessions.map((id) => (
              <li key={id}>
                <Link href={`/sessions/${id}`} className="mono">
                  {id}
                </Link>
              </li>
            ))}
          </ul>
        )}
      </div>
    </>
  );
}
