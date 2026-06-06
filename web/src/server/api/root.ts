import { bondsRouter } from "~/server/api/routers/bonds";
import { immunizationRouter } from "~/server/api/routers/immunization";
import { quantumRouter } from "~/server/api/routers/quantum";
import { createCallerFactory, createTRPCRouter } from "~/server/api/trpc";

/**
 * This is the primary router for your server.
 *
 * All routers added in /api/routers should be manually added here.
 */
export const appRouter = createTRPCRouter({
  bonds: bondsRouter,
  immunization: immunizationRouter,
  quantum: quantumRouter,
});

// export type definition of API
export type AppRouter = typeof appRouter;

/**
 * Create a server-side caller for the tRPC API.
 * @example
 * const trpc = createCaller(createContext);
 * const res = await trpc.bonds.search({ query: "" });
 */
export const createCaller = createCallerFactory(appRouter);
