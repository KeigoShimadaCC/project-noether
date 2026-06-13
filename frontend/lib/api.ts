// Typed client for the session API. The shapes mirror
// noether/orchestrator/view.py and noether/server/app.py exactly; the
// frontend renders them and never derives physics on its own.

export interface Question {
  id: string;
  question: string;
  kind: string;
  options: string[];
  resolution: string | null;
}

export interface NprObject {
  name: string;
  kind: string;
  role: string;
  definition_tex: string | null;
}

export interface SessionPayload {
  session_id: string;
  state: string;
  well_posed: boolean;
  action: { measure_tex: string; lagrangian_tex: string | null };
  objects: NprObject[];
  questions: Question[];
  events: { state: string; detail: string }[];
}

export interface DefinitionProposal {
  id: string;
  symbol: string;
  symbol_tex: string;
  meaning_tex: string;
  definition_tex: string;
  rationale: string;
}

export interface DefinitionsPayload {
  confirmed: false;
  note: string;
  proposals: DefinitionProposal[];
}

export interface PlanPayload {
  task_type: string;
  steps: { capability: string; description: string }[];
  verification: string[];
}

export interface BlockedPlan {
  blocked: true;
  questions: string[];
}

export interface FieldDerivation {
  wrt: string;
  kind: string;
  capability: string;
  result_tex: string | null;
  verified: boolean;
  checks: Record<string, string>;
  kernel_name: string;
  kernel_version: string;
  llm_name: string;
  llm_version: string;
  script: string;
  bundle_path: string | null;
  detail: string;
}

export interface DerivePayload {
  session_id: string;
  derivations: FieldDerivation[];
}

export interface Proposal {
  ambiguity_id: string;
  choice: string | null;
  rationale: string;
}

export interface ElicitPayload {
  confirmed: false;
  note: string;
  llm: { name: string; version: string };
  proposals: Proposal[];
}

export class ApiError extends Error {
  constructor(
    public status: number,
    public detail: unknown,
  ) {
    super(typeof detail === "string" ? detail : JSON.stringify(detail));
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`/api${path}`, {
    ...init,
    headers: { "Content-Type": "application/json", ...init?.headers },
  });
  const body = await response.json().catch(() => null);
  if (!response.ok) {
    throw new ApiError(response.status, body?.detail ?? response.statusText);
  }
  return body as T;
}

export const api = {
  health: () => request<{ status: string; kernels: Record<string, { available: boolean; version: string | null }> }>("/health"),
  listSessions: () => request<{ sessions: string[] }>("/sessions"),
  createSession: (lagrangian: string, measure?: string) =>
    request<SessionPayload>("/sessions", {
      method: "POST",
      body: JSON.stringify(measure ? { lagrangian, measure } : { lagrangian }),
    }),
  getSession: (id: string) => request<SessionPayload>(`/sessions/${id}`),
  resolve: (id: string, resolutions: Record<string, string>) =>
    request<SessionPayload>(`/sessions/${id}/resolve`, {
      method: "POST",
      body: JSON.stringify({ resolutions }),
    }),
  elicit: (id: string) => request<ElicitPayload>(`/sessions/${id}/elicit`, { method: "POST" }),
  definitions: (id: string) => request<DefinitionsPayload>(`/sessions/${id}/definitions`),
  adoptDefinitions: (id: string, accept: string[]) =>
    request<SessionPayload>(`/sessions/${id}/definitions`, {
      method: "POST",
      body: JSON.stringify({ accept }),
    }),
  plan: (id: string) => request<PlanPayload>(`/sessions/${id}/plan`),
  derive: (id: string, withRespectTo?: string[], kind: string = "eom") =>
    request<DerivePayload>(`/sessions/${id}/derive`, {
      method: "POST",
      body: JSON.stringify({
        kind,
        ...(withRespectTo ? { with_respect_to: withRespectTo } : {}),
      }),
    }),
};
