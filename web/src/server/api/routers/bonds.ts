import { z } from "zod";

import { createTRPCRouter, publicProcedure } from "~/server/api/trpc";
import { searchBonds } from "~/server/lib/bonds";

export const bondsRouter = createTRPCRouter({
  search: publicProcedure
    .input(z.object({ query: z.string().max(64).default("") }))
    .query(({ input }) => searchBonds(input.query)),
});
