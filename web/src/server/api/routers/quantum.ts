import { TRPCError } from "@trpc/server";
import { z } from "zod";

import { createTRPCRouter, publicProcedure } from "~/server/api/trpc";
import { quantumCli } from "~/server/lib/quantum-cli";
import {
  budgetSnapshot,
  markInflight,
  releaseInflight,
  reserveQpuSlot,
} from "~/server/lib/ratelimit";

const runInput = z.object({
  size: z.union([z.literal(7), z.literal(15), z.literal(31)]),
  maturities: z.array(z.number().positive()).optional(),
});

export const quantumRouter = createTRPCRouter({
  budget: publicProcedure.query(() => budgetSnapshot()),

  // Submit to real IBM hardware. Rate-limited + budget-guarded.
  run: publicProcedure.input(runInput).mutation(async ({ input }) => {
    const decision = reserveQpuSlot();
    if (!decision.ok) {
      throw new TRPCError({
        code: "TOO_MANY_REQUESTS",
        message: decision.reason ?? "Rate limited",
      });
    }
    try {
      const res = await quantumCli.submit(input.size, input.maturities);
      markInflight(res.job_id);
      return res;
    } catch (e) {
      // Submission failed — the reservation already consumed budget; that's an
      // acceptable conservative behaviour (protects against retry storms).
      throw new TRPCError({
        code: "INTERNAL_SERVER_ERROR",
        message: e instanceof Error ? e.message : "submit failed",
      });
    }
  }),

  // Poll an IBM job. Free (no QPU cost). Releases the in-flight slot when DONE.
  poll: publicProcedure
    .input(
      runInput.extend({
        jobId: z.string().min(1),
      }),
    )
    .query(async ({ input }) => {
      const res = await quantumCli.result(
        input.jobId,
        input.size,
        input.maturities,
      );
      if (res.done) releaseInflight(input.jobId);
      return res;
    }),
});
