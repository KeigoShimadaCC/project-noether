/**
 * Component tests for the derivation card: gated/unverified badges, stale
 * badges, the teaching narration panel, the convention block in the
 * provenance tree, and the LaTeX export that includes conventions and
 * excludes gated/stale results.
 *
 * These test the display contract:
 *  VAL-GUIDE-011/015: verified/unverified/stale badges and teaching panel
 *  VAL-CROSS-006: gated badge + detail matches the CLI reason
 *  VAL-CROSS-011: gated result visible but excluded from .tex export
 *  VAL-CROSS-015: verified results export with convention block
 *  VAL-CROSS-018: provenance tree shows action/plan/kernel/checks/conventions/result
 *  VAL-CROSS-020: teaching is never folded into result_tex/checks/verified
 */

import { render, screen } from "@testing-library/react";
import "@testing-library/jest-dom";
import DerivationTree from "@/components/DerivationTree";
import ExportPanel from "@/components/ExportPanel";
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

const baseConventions: Record<string, string> = {
  convention_id: "noether-default-v1",
  signature: "mostly-plus",
  torsion_sign: "+1",
  nonmetricity_definition: "nabla-g",
  ricci_contraction: "first-third",
  contortion_sign: "+1",
  disformation_sign: "+1",
  K_sign: "+1 (K_{ij}=+nabla_i n_j expansion-positive)",
  foliation_normal: "n_mu=(-N,0,...,0) timelike",
  field_strength_definition: "exterior-derivative",
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
    conventions: { ...baseConventions },
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

// --- Cross-flow tests (VAL-CROSS-006/011/015/018/020) ---

describe("DerivationTree provenance tree: conventions node (VAL-CROSS-018)", () => {
  it("renders a conventions node when the derivation carries a convention block", () => {
    const d = makeDerivation({ conventions: { ...baseConventions } });
    const { container } = render(
      <DerivationTree derivation={d} action={baseAction} plan={basePlan} />,
    );

    const conventionsNode = container.querySelector(".dtree-node.conventions");
    expect(conventionsNode).toBeInTheDocument();
    // Convention ID is visible
    expect(screen.getByText("noether-default-v1")).toBeInTheDocument();
    // At least one convention entry is visible (signature)
    expect(screen.getByText("mostly-plus")).toBeInTheDocument();
  });

  it("hides the conventions node when conventions is empty", () => {
    const d = makeDerivation({ conventions: {} });
    const { container } = render(
      <DerivationTree derivation={d} action={baseAction} plan={basePlan} />,
    );

    const conventionsNode = container.querySelector(".dtree-node.conventions");
    expect(conventionsNode).not.toBeInTheDocument();
  });

  it("shows every check the kernel reported, including cross-checks", () => {
    const d = makeDerivation({
      checks: { residue_zero: "True", linearized_eom_match: "True", sympy_cross_check: "True" },
    });
    render(<DerivationTree derivation={d} action={baseAction} plan={basePlan} />);

    expect(screen.getByText("residue_zero")).toBeInTheDocument();
    expect(screen.getByText("linearized_eom_match")).toBeInTheDocument();
    expect(screen.getByText("sympy_cross_check")).toBeInTheDocument();
  });

  it("exposes action, plan, kernel, checks, conventions, and result in the tree", () => {
    const d = makeDerivation({ conventions: { ...baseConventions } });
    const { container } = render(
      <DerivationTree derivation={d} action={baseAction} plan={basePlan} />,
    );

    // Action node
    expect(container.querySelector(".dtree-label")).toHaveTextContent("action");
    // Plan node
    const labels = Array.from(container.querySelectorAll(".dtree-label"));
    expect(labels.some((l) => l.textContent?.includes("plan"))).toBe(true);
    // Kernel node
    expect(labels.some((l) => l.textContent?.includes("kernel"))).toBe(true);
    // Verification node
    expect(labels.some((l) => l.textContent === "verification")).toBe(true);
    // Conventions node
    expect(labels.some((l) => l.textContent === "conventions")).toBe(true);
    // Result node
    expect(labels.some((l) => l.textContent === "result")).toBe(true);
  });
});

describe("DerivationTree gated badge detail (VAL-CROSS-006)", () => {
  it("shows the kernel reason on the unverified badge matching the CLI detail", () => {
    const reason = "unverified: the kernel computed a nonzero residue";
    const d = makeDerivation({ verified: false, detail: reason });
    render(<DerivationTree derivation={d} action={baseAction} plan={basePlan} />);

    expect(screen.getByText("unverified")).toHaveClass("badge", "unverified");
    expect(screen.getByText(new RegExp(reason.substring(0, 30)))).toBeInTheDocument();
  });

  it("distinguishes a verified result from a gated one by badge and detail", () => {
    const dVerified = makeDerivation({
      verified: true,
      detail: "kernel confirmed the variation matches the candidate equation",
    });
    const { unmount } = render(
      <DerivationTree derivation={dVerified} action={baseAction} plan={basePlan} />,
    );
    expect(screen.getByText("kernel-verified")).toHaveClass("badge", "verified");
    expect(screen.getByText(/kernel confirmed/)).toBeInTheDocument();
    unmount();

    const dGated = makeDerivation({
      verified: false,
      detail: "unverified: nonzero residue, model candidate does not match derivation",
    });
    render(<DerivationTree derivation={dGated} action={baseAction} plan={basePlan} />);
    expect(screen.getByText("unverified")).toHaveClass("badge", "unverified");
    expect(screen.getByText(/nonzero residue/)).toBeInTheDocument();
  });
});

describe("DerivationTree teaching never folded into result/checks (VAL-CROSS-020)", () => {
  it("teaching text does not appear in checks or result_tex", () => {
    const teachingText = "The metric-affine geometry introduces torsion coupling.";
    const d = makeDerivation({
      teaching: teachingText,
      result_tex: "R_{\\mu\\nu} - \\frac12 g_{\\mu\\nu} R = 0",
      checks: { residue_zero: "True" },
    });
    const { container } = render(
      <DerivationTree derivation={d} action={baseAction} plan={basePlan} />,
    );

    // Teaching is in its own node
    const teachingNode = container.querySelector(".dtree-node.teaching");
    expect(teachingNode).toBeInTheDocument();
    expect(teachingNode).toHaveTextContent(teachingText);

    // Teaching text does NOT appear in the result node
    const resultNode = container.querySelector(".dtree-node.result");
    expect(resultNode?.textContent).not.toContain("metric-affine geometry introduces torsion");

    // Teaching text does NOT appear in the verification/checks node
    const allNodes = container.querySelectorAll(".dtree-node");
    const checksNode = Array.from(allNodes).find((n) => {
      const label = n.querySelector(".dtree-label");
      return label?.textContent === "verification";
    });
    expect(checksNode?.textContent).not.toContain("metric-affine geometry introduces torsion");
  });
});

describe("ExportPanel LaTeX export (VAL-CROSS-011/015)", () => {
  it("excludes gated (unverified) results from the .tex export", () => {
    const verified = makeDerivation({ verified: true, result_id: "r-1" });
    const gated = makeDerivation({
      verified: false,
      result_id: "r-2",
      result_tex: "should_not_appear",
      detail: "unverified: nonzero residue",
    });
    render(
      <ExportPanel action={baseAction} derivations={[verified, gated]} staleResultIds={[]} />,
    );

    const textarea = screen.getByRole("textbox");
    const tex = (textarea as HTMLTextAreaElement).value;
    expect(tex).toContain("R_{");  // verified result appears
    expect(tex).not.toContain("should_not_appear");  // gated result excluded
  });

  it("excludes stale results from the .tex export", () => {
    const stale = makeDerivation({ verified: true, result_id: "r-stale", result_tex: "stale_eq" });
    const fresh = makeDerivation({ verified: true, result_id: "r-fresh", result_tex: "fresh_eq" });
    render(
      <ExportPanel action={baseAction} derivations={[stale, fresh]} staleResultIds={["r-stale"]} />,
    );

    const textarea = screen.getByRole("textbox");
    const tex = (textarea as HTMLTextAreaElement).value;
    expect(tex).not.toContain("stale_eq");
    expect(tex).toContain("fresh_eq");
  });

  it("includes the named convention block in the .tex export", () => {
    const d = makeDerivation({ conventions: { ...baseConventions } });
    render(
      <ExportPanel action={baseAction} derivations={[d]} staleResultIds={[]} />,
    );

    const textarea = screen.getByRole("textbox");
    const tex = (textarea as HTMLTextAreaElement).value;
    // Convention ID appears in the header comment
    expect(tex).toContain("noether-default-v1");
    // Individual convention entries appear as comment lines
    expect(tex).toContain("signature: mostly-plus");
    expect(tex).toContain("ricci_contraction: first-third");
  });

  it("shows nothing when all results are gated or stale", () => {
    const gated = makeDerivation({ verified: false, result_id: "r-gated" });
    const { container } = render(
      <ExportPanel action={baseAction} derivations={[gated]} staleResultIds={[]} />,
    );

    const textarea = container.querySelector("textarea.export-tex");
    expect(textarea).not.toBeInTheDocument();
  });
});
