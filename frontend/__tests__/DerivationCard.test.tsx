/**
 * Component tests for the derivation card: gated/unverified badges, stale
 * badges, and the teaching narration panel.
 *
 * These test the display contract (VAL-GUIDE-011, VAL-GUIDE-015):
 *  - A verified derivation shows a "kernel-verified" badge
 *  - A gated (unverified) derivation shows an "unverified" badge with
 *    different styling and the gated reason (detail)
 *  - A stale result shows a "stale" badge
 *  - Teaching renders in its own labeled region, visually distinct from
 *    result LaTeX and verification checks
 */

import { render, screen } from "@testing-library/react";
import "@testing-library/jest-dom";
import DerivationTree from "@/components/DerivationTree";
import type { FieldDerivation, SessionPayload, PlanPayload } from "@/lib/api";

// Minimal fixtures

const baseAction: SessionPayload["action"] = {
  measure_tex: "d^4x \\sqrt{-g}",
  lagrangian_tex: "R",
};

const basePlan: PlanPayload = {
  task_type: "vary",
  steps: [{ capability: "independent-connection", description: "vary g, Gamma" }],
  verification: ["residue"],
};

function makeDerivation(overrides: Partial<FieldDerivation> = {}): FieldDerivation {
  return {
    wrt: "g",
    kind: "eom",
    capability: "inDEPENDENT_CONNECTION" as any,
    result_id: "r-abc",
    result_tex: "R_{\\mu\\nu} - \\frac12 g_{\\mu\\nu} R = 0",
    verified: true,
    checks: { residue_zero: "True" },
    kernel_name: "cadabra",
    kernel_version: "2.5.15",
    llm_name: "codex",
    llm_version: "0.1",
    script: "# script",
    bundle_path: null,
    detail: "residue check passed",
    teaching: "",
    ...overrides,
  };
}

describe("DerivationTree verdict badges", () => {
  it("renders a kernel-verified badge when verified=true", () => {
    const d = makeDerivation({ verified: true, detail: "residue check passed" });
    render(<DerivationTree derivation={d} action={baseAction} plan={basePlan} />);

    const badge = screen.getByText("kernel-verified");
    expect(badge).toBeInTheDocument();
    expect(badge).toHaveClass("badge", "verified");
  });

  it("renders an unverified badge with the gated reason when verified=false", () => {
    const d = makeDerivation({
      verified: false,
      detail: "needs normal-ordering (SortCovDs unavailable)",
    });
    render(<DerivationTree derivation={d} action={baseAction} plan={basePlan} />);

    const badge = screen.getByText("unverified");
    expect(badge).toBeInTheDocument();
    expect(badge).toHaveClass("badge", "unverified");

    // The gated reason must be visible
    expect(
      screen.getByText(/needs normal-ordering/),
    ).toBeInTheDocument();
  });

  it("renders a stale badge when the result is stale", () => {
    const d = makeDerivation({ verified: true, detail: "residue check passed" });
    render(
      <DerivationTree
        derivation={d}
        action={baseAction}
        plan={basePlan}
        stale={true}
      />,
    );

    const badge = screen.getByText("stale");
    expect(badge).toBeInTheDocument();
    expect(badge).toHaveClass("badge", "stale");
  });
});

describe("DerivationTree teaching panel", () => {
  it("renders teaching in its own labeled region separate from result and checks", () => {
    const d = makeDerivation({
      teaching:
        "The independent connection introduces torsion, which couples to spin. " +
        "Without torsion, the connection reduces to Levi-Civita.",
    });
    const { container } = render(
      <DerivationTree derivation={d} action={baseAction} plan={basePlan} />,
    );

    // Teaching panel has its own labeled region
    const teachingNode = container.querySelector(".dtree-node.teaching");
    expect(teachingNode).toBeInTheDocument();

    // The teaching label is visible
    expect(screen.getByText("teaching")).toBeInTheDocument();

    // Teaching content is present
    expect(screen.getByText(/independent connection introduces torsion/)).toBeInTheDocument();

    // Teaching is NOT inside the result node
    const resultNode = container.querySelector(".dtree-node.result");
    expect(resultNode).not.toContainElement(
      container.querySelector(".teaching-body"),
    );

    // Teaching is NOT inside the verification node
    // (The verification node is the dtree-node containing the checks list;
    //  it does not contain the teaching-body element)
    const allDtreeNodes = container.querySelectorAll(".dtree-node");
    const verificationNode = Array.from(allDtreeNodes).find((node) => {
      const label = node.querySelector(".dtree-label");
      return label?.textContent === "verification";
    }) as HTMLElement | null;
    const teachingBody = container.querySelector(".teaching-body") as HTMLElement | null;
    expect(verificationNode).not.toContainElement(teachingBody);
  });

  it("hides the teaching panel when teaching is empty", () => {
    const d = makeDerivation({ teaching: "" });
    const { container } = render(
      <DerivationTree derivation={d} action={baseAction} plan={basePlan} />,
    );

    const teachingNode = container.querySelector(".dtree-node.teaching");
    expect(teachingNode).not.toBeInTheDocument();
  });

  it("labels teaching as reasoned, not kernel-verified", () => {
    const d = makeDerivation({
      teaching: "Torsion enables spin coupling.",
    });
    render(<DerivationTree derivation={d} action={baseAction} plan={basePlan} />);

    // The teaching panel is labeled as "reasoned" (not kernel-verified)
    expect(screen.getByText(/reasoned, not kernel-verified/i)).toBeInTheDocument();
  });
});
