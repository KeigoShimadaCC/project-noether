"use client";

import { useState } from "react";
import Latex from "@/components/Latex";
import type { FieldDerivation, PlanPayload, SessionPayload } from "@/lib/api";

// A self-contained provenance tree for one derivation: the action it started
// from, the plan that shaped it, the kernel that ran, every verification check
// the kernel reported, and the result it confirmed. This renders data the
// server already returned; it computes no physics.

export function headingFor(d: FieldDerivation): string {
  if (d.kind === "adm") return d.wrt;
  if (d.kind === "perturbation") return `S₂[${d.wrt}] (quadratic action)`;
  return `δS / δ${d.wrt} = 0`;
}

export default function DerivationTree({
  derivation: d,
  action,
  plan,
}: {
  derivation: FieldDerivation;
  action: SessionPayload["action"];
  plan: PlanPayload | null;
}) {
  const [openScript, setOpenScript] = useState(false);
  const checks = Object.entries(d.checks ?? {});

  return (
    <div className="dtree">
      <div className="dtree-node">
        <span className="dtree-label">action</span>
        <div className="dtree-body">
          <Latex
            tex={`S = \\int ${action.measure_tex}\\,\\left(${action.lagrangian_tex ?? ""}\\right)`}
            block
          />
        </div>
      </div>

      {plan && (
        <div className="dtree-node">
          <span className="dtree-label">plan ({plan.task_type})</span>
          <ol className="dtree-body plan-steps">
            {plan.steps.map((step, index) => (
              <li key={index}>
                <span className="badge resolved">{step.capability}</span> {step.description}
              </li>
            ))}
          </ol>
        </div>
      )}

      <div className="dtree-node">
        <span className="dtree-label">
          {d.kernel_name ? `kernel: ${d.kernel_name} ${d.kernel_version}` : "kernel"}
        </span>
        <div className="dtree-body">
          {d.script ? (
            <>
              <button className="secondary" onClick={() => setOpenScript(!openScript)}>
                {openScript ? "hide kernel script" : "show kernel script"}
              </button>
              {openScript && <pre className="script">{d.script}</pre>}
            </>
          ) : (
            <span className="note">no script: verified by component evaluation</span>
          )}
        </div>
      </div>

      <div className="dtree-node">
        <span className="dtree-label">verification</span>
        <ul className="dtree-body checks">
          {checks.length === 0 ? (
            <li className="note">no checks reported</li>
          ) : (
            checks.map(([name, value]) => (
              <li key={name}>
                <span className={`badge ${value === "True" ? "resolved" : "error"}`}>
                  {value === "True" ? "PASS" : "FAIL"}
                </span>{" "}
                <span className="mono">{name}</span>
              </li>
            ))
          )}
        </ul>
      </div>

      <div className="dtree-node result">
        <span className="dtree-label">result</span>
        <div className="dtree-body">
          <div className="defn-row">
            <span className="mono">{headingFor(d)}</span>
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
            {d.detail}. computed by {d.kernel_name} {d.kernel_version}
            {d.llm_name ? `; script by ${d.llm_name} ${d.llm_version}` : ""}.
          </div>
        </div>
      </div>
    </div>
  );
}
