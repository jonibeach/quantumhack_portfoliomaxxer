import { z } from "zod";

import { createTRPCRouter, publicProcedure } from "~/server/api/trpc";
import { quantumCli } from "~/server/lib/quantum-cli";

const SIZES = [7, 15, 31] as const;

const baseInput = z.object({
  size: z.union([z.literal(7), z.literal(15), z.literal(31)]),
  // Real maturities (years) for the selected bonds; length must equal size.
  maturities: z.array(z.number().positive()).optional(),
});

function validate(input: z.infer<typeof baseInput>) {
  if (input.maturities && input.maturities.length !== input.size) {
    throw new Error(
      `expected ${input.size} maturities, got ${input.maturities.length}`,
    );
  }
}

export const immunizationRouter = createTRPCRouter({
  sizes: publicProcedure.query(() => SIZES),

  instance: publicProcedure.input(baseInput).query(({ input }) => {
    validate(input);
    return quantumCli.instance(input.size, input.maturities);
  }),

  // Instant, free, unlimited simulator preview.
  preview: publicProcedure.input(baseInput).mutation(({ input }) => {
    validate(input);
    return quantumCli.simulate(input.size, input.maturities);
  }),
});
