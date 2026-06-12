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

export interface SessionPayload {
  session_id: string;
  state: string;
  well_posed: boolean;
  action: { measure_tex: string; lagrangian_tex: string | null };
  objects: { name: string; kind: string; role: string }[];
  questions: Question[];
  events: { state: string; detail: string }[];
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
  plan: (id: string) => request<PlanPayload>(`/sessions/${id}/plan`),
};
