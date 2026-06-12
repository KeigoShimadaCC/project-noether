/**
 * The browser only ever talks to Next; /api/* is proxied to the FastAPI
 * session server so the frontend stays same-origin and holds no physics
 * state (docs/02_TECH_SPEC.md section 2).
 */
const apiUrl = process.env.NOETHER_API_URL ?? "http://127.0.0.1:8754";

/** @type {import('next').NextConfig} */
const nextConfig = {
  async rewrites() {
    return [{ source: "/api/:path*", destination: `${apiUrl}/:path*` }];
  },
};

export default nextConfig;
