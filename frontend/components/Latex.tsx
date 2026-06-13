"use client";

import katex from "katex";
import { useMemo } from "react";

function escapeHtml(text: string): string {
  return text.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

export default function Latex({ tex, block = false }: { tex: string; block?: boolean }) {
  const html = useMemo(() => {
    // throwOnError: false renders bad fragments in red, which is the right
    // feedback while the user is mid-keystroke; the catch covers the rare
    // KaTeX errors that throw regardless.
    try {
      return katex.renderToString(tex, {
        displayMode: block,
        throwOnError: false,
        strict: false,
      });
    } catch {
      return `<code>${escapeHtml(tex)}</code>`;
    }
  }, [tex, block]);
  const Tag = block ? "div" : "span";
  return <Tag className="latex" dangerouslySetInnerHTML={{ __html: html }} />;
}
