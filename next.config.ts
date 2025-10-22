import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  eslint: {
    ignoreDuringBuilds: true, // allows build even if ESLint errors exist
  },
  typescript: {
    ignoreBuildErrors: true, // allows build even if TS errors exist
  },
};

export default nextConfig;
