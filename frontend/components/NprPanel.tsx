"use client";

import Latex from "@/components/Latex";
import type { SessionPayload } from "@/lib/api";

// The side panel is the UI expression of the NPR: what the problem IS right
// now, with every assumption visible and clickable (docs/02_TECH_SPEC.md).

export default function NprPanel({
  session,
  onReopen,
}: {
  session: SessionPayload;
  onReopen: (questionId: string) => void;
}) {
  const { action, objects, questions, events } = session;
  return (
    <div>
      <div className="card">
        <h2>Problem definition</h2>
        <dl className="kv">
          <dt>Session</dt>
          <dd className="mono">{session.session_id}</dd>
          <dt>State</dt>
          <dd>
            {session.state}{" "}
            <span className={`badge ${session.well_posed ? "resolved" : "open"}`}>
              {session.well_posed ? "well posed" : "questions open"}
            </span>
          </dd>
          <dt>Action</dt>
          <dd>
            <Latex
              tex={`\\int ${action.measure_tex} \\, \\left( ${action.lagrangian_tex ?? "\\mathcal{L}"} \\right)`}
              block
            />
          </dd>
        </dl>
        <h2 style={{ marginTop: "0.8rem" }}>Objects</h2>
        <ul className="object-list">
          {objects.map((object) => (
            <li key={object.name}>
              <Latex tex={object.name} /> <span className="qmeta">{object.kind}, {object.role}</span>
            </li>
          ))}
        </ul>
      </div>

      <div className="card">
        <h2>Assumptions</h2>
        {questions.map((question) => (
          <div key={question.id} style={{ marginBottom: "0.5rem" }}>
            <span className={`badge ${question.resolution ? "resolved" : "open"}`}>
              {question.resolution ? "resolved" : "open"}
            </span>{" "}
            <span className="mono">{question.id}</span>
            {question.resolution && (
              <>
                {" "}
                = <span className="mono">{question.resolution}</span>{" "}
                <button className="secondary" onClick={() => onReopen(question.id)}>
                  change
                </button>
              </>
            )}
          </div>
        ))}
        <p className="note">
          Changing an answer creates a new problem version; results computed
          against the old one are marked stale, never silently recomputed.
        </p>
      </div>

      <div className="card">
        <h2>History</h2>
        <ol className="timeline">
          {events.map((event, index) => (
            <li key={index}>
              <span className="mono">[{event.state}]</span> {event.detail}
            </li>
          ))}
        </ol>
      </div>
    </div>
  );
}
