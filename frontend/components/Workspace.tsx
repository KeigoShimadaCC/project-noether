"use client";

import { useCallback, useEffect, useState } from "react";
import Latex from "@/components/Latex";
import NprPanel from "@/components/NprPanel";
import {
  api,
  ApiError,
  type DefinitionProposal,
  type FieldDerivation,
  type PlanPayload,
  type Proposal,
  type Question,
  type SessionPayload,
} from "@/lib/api";

// The model may PROPOSE answers (elicit); nothing is applied until the human
// confirms each one, and confirmations are validated against the listed
// options server-side. The contract lives in the API; this UI only renders it.

function QuestionCard({
  question,
  proposal,
  busy,
  onResolve,
}: {
  question: Question;
  proposal?: Proposal;
  busy: boolean;
  onResolve: (choice: string) => void;
}) {
  const [freeform, setFreeform] = useState("");
  return (
    <div className="card">
      <h2>
        {question.question}
        <span className="qmeta">
          {question.id} ({question.kind})
        </span>
      </h2>
      <div className="option-list">
        {question.options.map((option) => (
          <button key={option} disabled={busy} onClick={() => onResolve(option)}>
            {option}
          </button>
        ))}
      </div>
      {proposal?.choice && (
        <div className="proposal">
          <div>
            model proposes <span className="mono">{proposal.choice}</span>{" "}
            <button disabled={busy} onClick={() => onResolve(proposal.choice as string)}>
              accept
            </button>
          </div>
          {proposal.rationale && <div className="rationale">{proposal.rationale}</div>}
        </div>
      )}
      <div className="freeform">
        <input
          type="text"
          value={freeform}
          placeholder="free-form answer (you are the authority)"
          onChange={(event) => setFreeform(event.target.value)}
        />
        <button
          className="secondary"
          disabled={busy || !freeform.trim()}
          onClick={() => onResolve(freeform.trim())}
        >
          record
        </button>
      </div>
      {freeform.includes("\\") && (
        <div className="preview">
          <span className="note">Live preview</span>
          <Latex tex={freeform} block />
        </div>
      )}
    </div>
  );
}

function NotationCard({
  proposals,
  busy,
  onAdopt,
}: {
  proposals: DefinitionProposal[];
  busy: boolean;
  onAdopt: (id: string) => void;
}) {
  if (proposals.length === 0) return null;
  return (
    <div className="card">
      <h2>Suggested notation</h2>
      <p className="note">
        Readability shorthands for the derivatives of your function couplings.
        These are definitions, not results; adopt the ones you like.
      </p>
      {proposals.map((proposal) => (
        <div key={proposal.id} className="proposal">
          <div className="defn-row">
            <Latex tex={proposal.definition_tex} />
            <button disabled={busy} onClick={() => onAdopt(proposal.id)}>
              adopt
            </button>
          </div>
          <div className="rationale">{proposal.rationale}</div>
        </div>
      ))}
    </div>
  );
}

function DerivationPanel({ sessionId }: { sessionId: string }) {
  const [results, setResults] = useState<FieldDerivation[] | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [openScript, setOpenScript] = useState<string | null>(null);

  async function run() {
    setBusy(true);
    setError(null);
    try {
      const payload = await api.derive(sessionId);
      setResults(payload.derivations);
    } catch (err) {
      setError(
        err instanceof ApiError && err.status === 503
          ? "The server needs both the Cadabra kernel and an agent CLI to derive. Neither physics nor the answer is guessed without them."
          : err instanceof ApiError && typeof err.detail === "string"
            ? err.detail
            : "derivation failed",
      );
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="card">
      <h2>Equations of motion</h2>
      <p className="note">
        Noether parameterizes a Cadabra script for this action, runs it in the
        kernel, and trusts the result only if the kernel confirms it. Results it
        cannot confirm are shown as unverified, never as truth. This can take a
        moment.
      </p>
      <button disabled={busy} onClick={run}>
        {busy ? "Deriving..." : results ? "Re-derive" : "Derive"}
      </button>
      {error && <div className="error-box">{error}</div>}
      {results?.map((d) => (
        <div key={d.wrt} className="proposal derivation">
          <div className="defn-row">
            <span className="mono">
              δS / δ{d.wrt} = 0
            </span>
            <span className={`badge ${d.verified ? "resolved" : "error"}`}>
              {d.verified ? "kernel-verified" : "unverified"}
            </span>
          </div>
          {d.result_tex ? (
            <div className="eom">
              <Latex tex={d.result_tex} block />
            </div>
          ) : (
            <p className="note">the kernel returned no expression</p>
          )}
          <div className="rationale">
            {d.detail}. computed by {d.kernel_name} {d.kernel_version}; script by{" "}
            {d.llm_name} {d.llm_version}.
          </div>
          <button
            className="secondary"
            onClick={() => setOpenScript(openScript === d.wrt ? null : d.wrt)}
          >
            {openScript === d.wrt ? "hide kernel script" : "show kernel script"}
          </button>
          {openScript === d.wrt && <pre className="script">{d.script}</pre>}
        </div>
      ))}
    </div>
  );
}

function PlanCard({ plan }: { plan: PlanPayload }) {
  return (
    <div className="card">
      <h2>
        Problem is well posed. Plan <span className="mono">({plan.task_type})</span>
      </h2>
      <ol className="plan-steps">
        {plan.steps.map((step, index) => (
          <li key={index}>
            <span className="badge resolved">{step.capability}</span> {step.description}
          </li>
        ))}
      </ol>
      <p>
        Verification ladder:{" "}
        {plan.verification.map((check) => (
          <span key={check} className="badge resolved" style={{ marginRight: "0.4rem" }}>
            {check}
          </span>
        ))}
      </p>
      <p className="note">
        Derivations for the supported task types run through the eval commands
        (noether eval1 .. eval5, eval1s, eval3s), which write full provenance
        bundles: kernel scripts, assumptions, and every check result.
      </p>
    </div>
  );
}

export default function Workspace({ sessionId }: { sessionId: string }) {
  const [session, setSession] = useState<SessionPayload | null>(null);
  const [plan, setPlan] = useState<PlanPayload | null>(null);
  const [proposals, setProposals] = useState<Record<string, Proposal>>({});
  const [proposalSource, setProposalSource] = useState<string | null>(null);
  const [definitions, setDefinitions] = useState<DefinitionProposal[]>([]);
  const [reopened, setReopened] = useState<Set<string>>(new Set());
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const refreshDefinitions = useCallback(async (id: string) => {
    try {
      const payload = await api.definitions(id);
      setDefinitions(payload.proposals);
    } catch {
      setDefinitions([]);
    }
  }, []);

  const refreshPlan = useCallback(async (payload: SessionPayload) => {
    if (!payload.well_posed) {
      setPlan(null);
      return;
    }
    try {
      setPlan(await api.plan(payload.session_id));
    } catch (err) {
      setPlan(null);
      setError(err instanceof ApiError ? err.message : "plan request failed");
    }
  }, []);

  useEffect(() => {
    api
      .getSession(sessionId)
      .then(async (payload) => {
        setSession(payload);
        await refreshPlan(payload);
        await refreshDefinitions(payload.session_id);
      })
      .catch((err) =>
        setError(
          err instanceof ApiError && err.status === 404
            ? `No session ${sessionId}`
            : "is the API server running? (noether serve)",
        ),
      );
  }, [sessionId, refreshPlan, refreshDefinitions]);

  async function adoptDefinition(definitionId: string) {
    setBusy(true);
    setError(null);
    try {
      const payload = await api.adoptDefinitions(sessionId, [definitionId]);
      setSession(payload);
      await refreshDefinitions(payload.session_id);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "could not adopt notation");
    } finally {
      setBusy(false);
    }
  }

  async function resolve(questionId: string, choice: string) {
    setBusy(true);
    setError(null);
    try {
      const payload = await api.resolve(sessionId, { [questionId]: choice });
      setSession(payload);
      setProposals((current) => {
        const next = { ...current };
        delete next[questionId];
        return next;
      });
      setReopened((current) => {
        const next = new Set(current);
        next.delete(questionId);
        return next;
      });
      await refreshPlan(payload);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "resolve failed");
    } finally {
      setBusy(false);
    }
  }

  async function propose() {
    setBusy(true);
    setError(null);
    try {
      const payload = await api.elicit(sessionId);
      const byId: Record<string, Proposal> = {};
      for (const proposal of payload.proposals) {
        if (proposal.choice !== null) byId[proposal.ambiguity_id] = proposal;
      }
      setProposals(byId);
      setProposalSource(`${payload.llm.name} ${payload.llm.version}`);
    } catch (err) {
      setError(
        err instanceof ApiError && err.status === 503
          ? "No agent CLI detected on the server; answer directly instead."
          : err instanceof ApiError
            ? err.message
            : "elicitation failed",
      );
    } finally {
      setBusy(false);
    }
  }

  if (!session) {
    return error ? <div className="error-box">{error}</div> : <p className="note">Loading...</p>;
  }

  const openQuestions = session.questions.filter(
    (question) => question.resolution === null || reopened.has(question.id),
  );

  return (
    <div className="workspace">
      <div>
        {error && <div className="error-box">{error}</div>}
        {openQuestions.length > 0 && (
          <div className="card">
            <h2>Clarifying questions</h2>
            <p className="note">
              Noether does not guess. Pick an option, record a free-form
              answer, or ask the model for proposals you then confirm.
            </p>
            <button className="secondary" disabled={busy} onClick={propose}>
              Ask the model to propose
            </button>
            {proposalSource && (
              <p className="note">
                Proposals from {proposalSource}; unconfirmed until you accept each one.
              </p>
            )}
          </div>
        )}
        {openQuestions.map((question) => (
          <QuestionCard
            key={question.id}
            question={question}
            proposal={proposals[question.id]}
            busy={busy}
            onResolve={(choice) => resolve(question.id, choice)}
          />
        ))}
        <NotationCard proposals={definitions} busy={busy} onAdopt={adoptDefinition} />
        {openQuestions.length === 0 && plan && <PlanCard plan={plan} />}
        {openQuestions.length === 0 && plan && <DerivationPanel sessionId={sessionId} />}
      </div>
      <NprPanel
        session={session}
        onReopen={(questionId) =>
          setReopened((current) => new Set(current).add(questionId))
        }
      />
    </div>
  );
}
