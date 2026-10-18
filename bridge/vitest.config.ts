import path from "node:path";
import { defineConfig } from "vitest/config";

const v = (p: string) => path.resolve(import.meta.dirname, "../vendor/openfront", p);

export default defineConfig({
  resolve: {
    alias: [
      { find: /^@openfront\/engine-api\/(.*)$/, replacement: v("packages/engine-api/src/$1") },
      { find: /^@openfront\/engine-lib\/(.*)$/, replacement: v("packages/engine-lib/src/$1") },
      { find: /^@openfront\/engine\/(.*)$/, replacement: v("packages/engine/src/$1") },
      { find: /^@openfront\/shared\/(.*)$/, replacement: v("packages/shared/src/$1") },
      { find: /^@openfront\/zbin\/(.*)$/, replacement: v("packages/zbin/src/$1") },
      { find: /^resources\/(.*)$/, replacement: v("resources/$1") },
    ],
  },
  test: { environment: "node", include: ["tests/**/*.test.ts"], testTimeout: 120_000 },
});
