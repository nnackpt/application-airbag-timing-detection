import type { NextConfig } from "next";

const backend = process.env.FASTAPI_URL || "http://127.0.0.1:8000"

const nextConfig: NextConfig = {
  /* config options here */
  async rewrites() {
    return [
      {
        source: "/fastapi/:path*",
        destination: `${backend}/:path*`,
      }
    ]
  }
};

export default nextConfig;
