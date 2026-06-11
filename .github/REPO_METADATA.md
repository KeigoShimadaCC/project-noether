# GitHub repository metadata

No remote exists yet. When the repo is first pushed to GitHub, apply this
metadata with the command at the bottom (requires `gh` authenticated).

## Description (under 350 chars)

> Agentic symbolic-physics collaborator: write an action in LaTeX, answer a
> few sharp questions, get back verified field equations with full provenance.
> An LLM orchestrates; established CAS kernels (Cadabra2, SymPy, xAct)
> compute; a verification ladder checks every result.

## Topics

`computer-algebra`, `symbolic-computation`, `general-relativity`,
`field-theory`, `tensor-calculus`, `physics`, `llm-agents`, `cadabra`,
`sympy`, `latex`

## Settings

- Default branch: `main`
- Issues: on. Wiki: off (docs live in-repo under `docs/`).
- License: not chosen yet (decide before making the repo public; note that
  Cadabra2 is GPL-3.0 and is invoked as a subprocess, not linked).

## Apply command

```sh
gh repo edit \
  --description "Agentic symbolic-physics collaborator: write an action in LaTeX, answer a few sharp questions, get back verified field equations with full provenance. An LLM orchestrates; established CAS kernels (Cadabra2, SymPy, xAct) compute; a verification ladder checks every result." \
  --add-topic computer-algebra --add-topic symbolic-computation \
  --add-topic general-relativity --add-topic field-theory \
  --add-topic tensor-calculus --add-topic physics --add-topic llm-agents \
  --add-topic cadabra --add-topic sympy --add-topic latex \
  --enable-issues --enable-wiki=false
```
