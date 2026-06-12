"use client";

import katex from "katex";
import { useMemo } from "react";

export default function Latex({ tex, block = false }: { tex: string; block?: boolean }) {
  const html = useMemo(
    () =>
      katex.renderToString(tex, {
        displayMode: block,
        throwOnError: false,
        strict: false,
      }),
    [tex, block],
  );
  const Tag = block ? "div" : "span";
  return <Tag className="latex" dangerouslySetInnerHTML={{ __html: html }} />;
}
