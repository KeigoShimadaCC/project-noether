import type { Metadata } from "next";
import Link from "next/link";
import "katex/dist/katex.min.css";
import "./globals.css";

export const metadata: Metadata = {
  title: "Noether",
  description:
    "Agentic symbolic-physics collaborator: LaTeX action in, verified field equations out",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>
        <header className="topbar">
          <Link href="/" className="brand">
            Noether
          </Link>
          <span className="tagline">action in, verified equations out</span>
        </header>
        <main>{children}</main>
      </body>
    </html>
  );
}
