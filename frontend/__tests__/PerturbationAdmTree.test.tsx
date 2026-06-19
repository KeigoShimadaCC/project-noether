/**
 * Component tests for DerivationTree rendering perturbation and ADM derivation
 * trees. These mirror the existing EOM tree tests in DerivationCard.test.tsx,
 * providing regression coverage for VAL-CROSS-019 (perturbation/ADM
 * reachability in the web UI).
 *
 * Scope: unit-level rendering assertions only. No component behavior changes.
 * If a rendering gap is found, return to the orchestrator rather than
 * expanding scope.
 */

import { render, screen } from "@testing-library/react";
import "@testing-library/jest-dom";
import DerivationTree from "@/components/DerivationTree";
import type { FieldDerivation, SessionPayload, PlanPayload } from "@/lib/api";

// --- Shared fixtures ---

const baseAction: SessionPayload["action"] = {
  measure_tex: "d^4x \\sqrt{-g}",
  lagrangian_tex: "R",
};

const perturbationPlan: PlanPayload = {
  task_type: "perturb",
  steps: [{ capability: "perturb", description: "quadratic-action expansion for h" }],
  verification: ["residue", "linearized_eom_match"],
};

const admPlan: PlanPayload = {
  task_type: "adm",
  steps: [{ capability: "adm", description: "3+1 decomposition of the gravitational sector" }],
  verification: ["component_eval"],
};

const metricAffineConventions: Record<string, string> = {
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

// --- Perturbation derivation fixtures ---

function makePerturbationDerivation(
  overrides: Partial<FieldDerivation> = {},
): FieldDerivation {
  return {
    wrt: "h",
    kind: "perturbation",
    capability: "PERTURB" as any,
    result_id: "perturb-h-abc",
    result_tex:
      "S_2 = \\tfrac12 \\int d^4x\\,\\sqrt{-g}\\,"
      + "\\left[h_{\\mu\\nu} E^{\\mu\\nu\\rho\\sigma} h_{\\rho\\sigma}\\right]",
    verified: true,
    checks: { residue_zero: "True", linearized_eom_match: "True" },
    kernel_name: "cadabra",
    kernel_version: "2.5.15",
    llm_name: "codex",
    llm_version: "0.1",
    script: "# perturbation script",
    bundle_path: null,
    detail: "kernel confirmed the quadratic action reproduces the linearized equation",
    teaching: "",
    conventions: { ...metricAffineConventions },
    ...overrides,
  };
}

// --- ADM derivation fixtures ---

function makeAdmDerivation(
  overrides: Partial<FieldDerivation> = {},
): FieldDerivation {
  return {
    wrt: "Gauss-Codazzi split of the gravitational Lagrangian",
    kind: "adm",
    capability: "ADM" as any,
    result_id: "adm-split-abc",
    result_tex:
      "\\sqrt{-g}\\,R = N\\sqrt{h}\\left(R^{(3)}"
      + " + K_{ab}K^{ab} - K^{2}\\right)"
      + " - 2\\,\\partial_{\\mu}\\!\\left(\\sqrt{-g}\\,v^{\\mu}\\right)",
    verified: true,
    checks: {
      einstein_normal_normal: "True",
      einstein_normal_tangential: "True",
      extrinsic_curvature_identity: "True",
      lapse_euler_lagrange: "True",
    },
    kernel_name: "sympy",
    kernel_version: "1.14",
    llm_name: "",
    llm_version: "",
    script: "# component evaluation script",
    bundle_path: null,
    detail:
      "kernel confirmed the ADM split, both Einstein-tensor projections, the "
      + "extrinsic-curvature identity, and the lapse Euler-Lagrange equation on "
      + "an explicit 1+2 background",
    teaching: "",
    conventions: { ...metricAffineConventions },
    ...overrides,
  };
}

function makeAdmKijDerivation(
  overrides: Partial<FieldDerivation> = {},
): FieldDerivation {
  return {
    wrt: "extrinsic curvature convention",
    kind: "adm",
    capability: "ADM" as any,
    result_id: "adm-kij-abc",
    result_tex:
      "K_{ij} = +\\nabla_i n_j\\;\\text{(expansion-positive, }n_\\mu = (-N,0,\\ldots,0)\\text{)}",
    verified: true,
    checks: {
      einstein_normal_normal: "True",
      einstein_normal_tangential: "True",
      extrinsic_curvature_identity: "True",
      lapse_euler_lagrange: "True",
    },
    kernel_name: "sympy",
    kernel_version: "1.14",
    llm_name: "",
    llm_version: "",
    script: "# component evaluation script",
    bundle_path: null,
    detail:
      "kernel confirmed the ADM split, both Einstein-tensor projections, the "
      + "extrinsic-curvature identity, and the lapse Euler-Lagrange equation on "
      + "an explicit 1+2 background",
    teaching: "",
    conventions: { ...metricAffineConventions },
    ...overrides,
  };
}

function makeAdmConstraintDerivation(
  overrides: Partial<FieldDerivation> = {},
): FieldDerivation {
  return {
    wrt: "connection-sector constraints",
    kind: "adm",
    capability: "ADM" as any,
    result_id: "adm-const-abc",
    result_tex:
      "\\text{Primary: }\\delta S/\\delta\\Gamma\\;\\text{involves }K(T),L(Q)"
      + ";\\;\\text{Dirac chain closure requires action-specific analysis}",
    verified: false,
    checks: {
      einstein_normal_normal: "True",
      einstein_normal_tangential: "True",
      extrinsic_curvature_identity: "True",
      lapse_euler_lagrange: "True",
      affine_foliation_projection: "True",
      distortion_spatial_projection: "True",
    },
    kernel_name: "sympy",
    kernel_version: "1.14",
    llm_name: "",
    llm_version: "",
    script: "# component evaluation script",
    bundle_path: null,
    detail:
      "Dirac chain cannot be closed for the general metric-affine case with "
      + "non-metricity (Q != 0): the disformation L(Q) introduces additional "
      + "structure that requires action-specific analysis. Gated with a stated reason.",
    teaching:
      "The independent connection introduces contortion K(T) and disformation "
      + "L(Q) as extra degrees of freedom. When non-metricity is present, the "
      + "Dirac constraint chain cannot be closed in general because L(Q) "
      + "introduces structure that depends on the specific action.",
    conventions: { ...metricAffineConventions },
    ...overrides,
  };
}

// ===================================================================
// Perturbation derivation tree tests
// ===================================================================

describe("DerivationTree: perturbation derivation", () => {
  it("renders the perturbation-specific heading S₂[wrt]", () => {
    const d = makePerturbationDerivation();
    render(<DerivationTree derivation={d} action={baseAction} plan={perturbationPlan} />);

    // headingFor perturbation: S₂[h] (quadratic action)
    const heading = screen.getByText(/S₂\[h\]/);
    expect(heading).toBeInTheDocument();
    expect(heading.textContent).toContain("quadratic action");
  });

  it("renders both residue_zero and linearized_eom_match checks", () => {
    const d = makePerturbationDerivation();
    render(<DerivationTree derivation={d} action={baseAction} plan={perturbationPlan} />);

    expect(screen.getByText("residue_zero")).toBeInTheDocument();
    expect(screen.getByText("linearized_eom_match")).toBeInTheDocument();
    // Both checks are PASS
    const passes = screen.getAllByText("PASS");
    expect(passes.length).toBeGreaterThanOrEqual(2);
  });

  it("renders a kernel-verified badge when the perturbation is verified", () => {
    const d = makePerturbationDerivation({ verified: true });
    render(<DerivationTree derivation={d} action={baseAction} plan={perturbationPlan} />);

    const badge = screen.getByText("kernel-verified");
    expect(badge).toBeInTheDocument();
    expect(badge).toHaveClass("badge", "verified");
  });

  it("renders an unverified badge with the gated reason when the perturbation is gated", () => {
    const d = makePerturbationDerivation({
      verified: false,
      checks: { residue_zero: "True", linearized_eom_match: "False" },
      detail:
        "unverified: the residue vanished but the independent linearized-EOM "
        + "cross-check did not agree",
    });
    render(<DerivationTree derivation={d} action={baseAction} plan={perturbationPlan} />);

    const badge = screen.getByText("unverified");
    expect(badge).toBeInTheDocument();
    expect(badge).toHaveClass("badge", "unverified");
    // The gated reason must be visible
    expect(screen.getByText(/linearized-EOM cross-check/)).toBeInTheDocument();
  });

  it("renders the conventions node with metric-affine convention fields", () => {
    const d = makePerturbationDerivation({ conventions: { ...metricAffineConventions } });
    const { container } = render(
      <DerivationTree derivation={d} action={baseAction} plan={perturbationPlan} />,
    );

    const conventionsNode = container.querySelector(".dtree-node.conventions");
    expect(conventionsNode).toBeInTheDocument();
    // Perturbation-specific conventions are present
    expect(screen.getByText("noether-default-v1")).toBeInTheDocument();
    expect(screen.getByText("mostly-plus")).toBeInTheDocument();
  });

  it("renders the quadratic-action result LaTeX", () => {
    const d = makePerturbationDerivation();
    // The result node exists
    const { container } = render(
      <DerivationTree derivation={d} action={baseAction} plan={perturbationPlan} />,
    );
    const resultNode = container.querySelector(".dtree-node.result");
    expect(resultNode).toBeInTheDocument();
  });

  it("renders a teaching panel for a perturbation derivation with teaching", () => {
    const d = makePerturbationDerivation({
      teaching:
        "The metric-affine perturbation includes the connection fluctuation dG "
        + "alongside the metric fluctuation h, mixing metric and connection "
        + "degrees of freedom at quadratic order.",
    });
    const { container } = render(
      <DerivationTree derivation={d} action={baseAction} plan={perturbationPlan} />,
    );

    const teachingNode = container.querySelector(".dtree-node.teaching");
    expect(teachingNode).toBeInTheDocument();
    expect(screen.getByText(/connection fluctuation dG/)).toBeInTheDocument();
    // Labeled as reasoned, not kernel-verified
    expect(screen.getByText(/reasoned, not kernel-verified/i)).toBeInTheDocument();
  });

  it("exposes action, plan, kernel, checks, conventions, and result in the perturbation tree", () => {
    const d = makePerturbationDerivation({ conventions: { ...metricAffineConventions } });
    const { container } = render(
      <DerivationTree derivation={d} action={baseAction} plan={perturbationPlan} />,
    );

    const labels = Array.from(container.querySelectorAll(".dtree-label"));
    expect(labels.some((l) => l.textContent?.includes("action"))).toBe(true);
    expect(labels.some((l) => l.textContent?.includes("plan"))).toBe(true);
    expect(labels.some((l) => l.textContent?.includes("kernel"))).toBe(true);
    expect(labels.some((l) => l.textContent === "verification")).toBe(true);
    expect(labels.some((l) => l.textContent === "conventions")).toBe(true);
    expect(labels.some((l) => l.textContent === "result")).toBe(true);
  });

  it("renders a FAIL badge on a failing linearized_eom_match check", () => {
    const d = makePerturbationDerivation({
      checks: { residue_zero: "True", linearized_eom_match: "False" },
    });
    render(<DerivationTree derivation={d} action={baseAction} plan={perturbationPlan} />);

    // The linearized_eom_match check shows FAIL
    const checkItems = screen.getAllByText("linearized_eom_match");
    expect(checkItems.length).toBeGreaterThanOrEqual(1);
    // There should be a FAIL badge (the check value is "False")
    const fails = screen.getAllByText("FAIL");
    expect(fails.length).toBeGreaterThanOrEqual(1);
  });
});

// ===================================================================
// ADM derivation tree tests
// ===================================================================

describe("DerivationTree: ADM derivation (GR, metric-sector pieces)", () => {
  it("renders the ADM-specific heading (wrt as piece label)", () => {
    const d = makeAdmDerivation();
    render(<DerivationTree derivation={d} action={baseAction} plan={admPlan} />);

    // headingFor adm: just d.wrt, which is "Gauss-Codazzi split..."
    expect(screen.getByText(/Gauss-Codazzi split/)).toBeInTheDocument();
  });

  it("renders the K_ij (extrinsic curvature) piece", () => {
    const d = makeAdmKijDerivation();
    render(<DerivationTree derivation={d} action={baseAction} plan={admPlan} />);

    // The wrt label is the piece label
    expect(screen.getByText(/extrinsic curvature convention/)).toBeInTheDocument();
  });

  it("renders the SymPy component kernel name, not a model script", () => {
    const d = makeAdmDerivation();
    const { container } = render(
      <DerivationTree derivation={d} action={baseAction} plan={admPlan} />,
    );

    // Kernel label shows sympy (in the dtree-label span)
    const kernelLabel = container.querySelector(".dtree-label");
    const allLabels = Array.from(container.querySelectorAll(".dtree-label"));
    const kernelNode = allLabels.find((l) => l.textContent?.includes("kernel: sympy"));
    expect(kernelNode).toBeInTheDocument();
    expect(kernelNode?.textContent).toContain("sympy 1.14");
  });

  it("renders all ADM component checks", () => {
    const d = makeAdmDerivation();
    render(<DerivationTree derivation={d} action={baseAction} plan={admPlan} />);

    expect(screen.getByText("einstein_normal_normal")).toBeInTheDocument();
    expect(screen.getByText("einstein_normal_tangential")).toBeInTheDocument();
    expect(screen.getByText("extrinsic_curvature_identity")).toBeInTheDocument();
    expect(screen.getByText("lapse_euler_lagrange")).toBeInTheDocument();
  });

  it("renders a kernel-verified badge when the ADM split is verified", () => {
    const d = makeAdmDerivation({ verified: true });
    render(<DerivationTree derivation={d} action={baseAction} plan={admPlan} />);

    const badge = screen.getByText("kernel-verified");
    expect(badge).toBeInTheDocument();
    expect(badge).toHaveClass("badge", "verified");
  });

  it("renders the conventions node with K_sign and foliation_normal", () => {
    const d = makeAdmDerivation({ conventions: { ...metricAffineConventions } });
    const { container } = render(
      <DerivationTree derivation={d} action={baseAction} plan={admPlan} />,
    );

    const conventionsNode = container.querySelector(".dtree-node.conventions");
    expect(conventionsNode).toBeInTheDocument();
    // ADM-specific convention fields
    expect(screen.getByText(/K_sign/)).toBeInTheDocument();
    expect(screen.getByText(/foliation_normal/)).toBeInTheDocument();
  });
});

describe("DerivationTree: ADM derivation (metric-affine, gated constraint piece)", () => {
  it("renders an unverified badge with the Dirac-chain blocker detail", () => {
    const d = makeAdmConstraintDerivation();
    render(<DerivationTree derivation={d} action={baseAction} plan={admPlan} />);

    const badge = screen.getByText("unverified");
    expect(badge).toBeInTheDocument();
    expect(badge).toHaveClass("badge", "unverified");
    // The gated reason naming the Dirac chain blocker must be visible
    expect(screen.getByText(/Dirac chain cannot be closed/)).toBeInTheDocument();
  });

  it("renders the connection-sector constraint LaTeX", () => {
    const d = makeAdmConstraintDerivation();
    const { container } = render(
      <DerivationTree derivation={d} action={baseAction} plan={admPlan} />,
    );

    const resultNode = container.querySelector(".dtree-node.result");
    expect(resultNode).toBeInTheDocument();
  });

  it("renders a teaching panel with metric-affine ADM narration", () => {
    const d = makeAdmConstraintDerivation();
    const { container } = render(
      <DerivationTree derivation={d} action={baseAction} plan={admPlan} />,
    );

    const teachingNode = container.querySelector(".dtree-node.teaching");
    expect(teachingNode).toBeInTheDocument();
    // Teaching mentions L(Q) and the constraint chain (inside the teaching node)
    expect(teachingNode?.textContent).toContain("disformation L(Q)");
    // Labeled as reasoned, not kernel-verified
    expect(screen.getByText(/reasoned, not kernel-verified/i)).toBeInTheDocument();
  });

  it("renders both metric-sector and affine-sector checks on the constraint piece", () => {
    const d = makeAdmConstraintDerivation();
    render(<DerivationTree derivation={d} action={baseAction} plan={admPlan} />);

    // Metric checks
    expect(screen.getByText("einstein_normal_normal")).toBeInTheDocument();
    expect(screen.getByText("einstein_normal_tangential")).toBeInTheDocument();
    // Affine checks
    expect(screen.getByText("affine_foliation_projection")).toBeInTheDocument();
    expect(screen.getByText("distortion_spatial_projection")).toBeInTheDocument();
  });

  it("renders the conventions node with all metric-affine convention fields", () => {
    const d = makeAdmConstraintDerivation({ conventions: { ...metricAffineConventions } });
    const { container } = render(
      <DerivationTree derivation={d} action={baseAction} plan={admPlan} />,
    );

    const conventionsNode = container.querySelector(".dtree-node.conventions");
    expect(conventionsNode).toBeInTheDocument();
    // Core convention fields
    expect(screen.getByText("noether-default-v1")).toBeInTheDocument();
    // ADM-relevant fields
    expect(screen.getByText(/K_sign/)).toBeInTheDocument();
    expect(screen.getByText(/foliation_normal/)).toBeInTheDocument();
    // Metric-affine fields
    expect(screen.getByText(/contortion_sign/)).toBeInTheDocument();
    expect(screen.getByText(/disformation_sign/)).toBeInTheDocument();
  });

  it("teaching is separate from result and checks (VAL-CROSS-020 for ADM)", () => {
    const d = makeAdmConstraintDerivation();
    const { container } = render(
      <DerivationTree derivation={d} action={baseAction} plan={admPlan} />,
    );

    const teachingNode = container.querySelector(".dtree-node.teaching");
    const resultNode = container.querySelector(".dtree-node.result");
    expect(teachingNode).toBeInTheDocument();
    expect(resultNode).toBeInTheDocument();

    // Teaching text does NOT appear in the result node
    expect(resultNode?.textContent).not.toContain("disformation L(Q) introduces structure");

    // Teaching text does NOT appear in the verification/checks node
    const allNodes = container.querySelectorAll(".dtree-node");
    const checksNode = Array.from(allNodes).find((n) => {
      const label = n.querySelector(".dtree-label");
      return label?.textContent === "verification";
    });
    expect(checksNode?.textContent).not.toContain("disformation L(Q) introduces structure");
  });
});

describe("DerivationTree: ADM stale result badge", () => {
  it("renders a stale badge on an ADM derivation marked stale", () => {
    const d = makeAdmDerivation({ verified: true });
    render(
      <DerivationTree
        derivation={d}
        action={baseAction}
        plan={admPlan}
        stale={true}
      />,
    );

    const badge = screen.getByText("stale");
    expect(badge).toBeInTheDocument();
    expect(badge).toHaveClass("badge", "stale");
  });
});
