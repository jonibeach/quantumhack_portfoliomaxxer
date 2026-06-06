import "server-only";

/**
 * Bond universe for the picker. The DQI system only needs a MATURITY per
 * instrument (the maturity ladder is mapped onto the code's locator set), so we
 * curate sovereign benchmark tenors with their maturities; live yields (where a
 * Yahoo Finance index symbol exists) are a nice-to-have display enrichment.
 *
 * Yahoo's unofficial chart endpoint is fetched server-side (no API key), cached
 * briefly, and failures are tolerated — a Yahoo hiccup must never block a run.
 */

export interface CatalogBond {
  id: string;
  label: string;
  country: string;
  flag: string;
  maturityYears: number;
  /** Yahoo index symbol for a live yield, if one exists for this tenor. */
  yahooSymbol?: string;
}

export interface Bond extends CatalogBond {
  /** Live value from Yahoo (yield %, index points), or null if unavailable. */
  liveYield: number | null;
}

// Curated catalog. US Treasuries are the core (clean tenors); a few sovereigns
// added where a recognizable benchmark exists.
const CATALOG: CatalogBond[] = [
  { id: "us-1m", label: "US T-Bill 1M", country: "US", flag: "🇺🇸", maturityYears: 1 / 12, yahooSymbol: "^IRX" },
  { id: "us-3m", label: "US T-Bill 3M", country: "US", flag: "🇺🇸", maturityYears: 0.25, yahooSymbol: "^IRX" },
  { id: "us-6m", label: "US T-Bill 6M", country: "US", flag: "🇺🇸", maturityYears: 0.5 },
  { id: "us-1y", label: "US Treasury 1Y", country: "US", flag: "🇺🇸", maturityYears: 1 },
  { id: "us-2y", label: "US Treasury 2Y", country: "US", flag: "🇺🇸", maturityYears: 2 },
  { id: "us-3y", label: "US Treasury 3Y", country: "US", flag: "🇺🇸", maturityYears: 3 },
  { id: "us-5y", label: "US Treasury 5Y", country: "US", flag: "🇺🇸", maturityYears: 5, yahooSymbol: "^FVX" },
  { id: "us-7y", label: "US Treasury 7Y", country: "US", flag: "🇺🇸", maturityYears: 7 },
  { id: "us-10y", label: "US Treasury 10Y", country: "US", flag: "🇺🇸", maturityYears: 10, yahooSymbol: "^TNX" },
  { id: "us-20y", label: "US Treasury 20Y", country: "US", flag: "🇺🇸", maturityYears: 20 },
  { id: "us-30y", label: "US Treasury 30Y", country: "US", flag: "🇺🇸", maturityYears: 30, yahooSymbol: "^TYX" },
  // Other sovereigns (10Y benchmarks).
  { id: "de-10y", label: "German Bund 10Y", country: "DE", flag: "🇩🇪", maturityYears: 10 },
  { id: "gb-10y", label: "UK Gilt 10Y", country: "GB", flag: "🇬🇧", maturityYears: 10 },
  { id: "jp-10y", label: "Japan JGB 10Y", country: "JP", flag: "🇯🇵", maturityYears: 10 },
  { id: "fr-10y", label: "France OAT 10Y", country: "FR", flag: "🇫🇷", maturityYears: 10 },
  { id: "it-10y", label: "Italy BTP 10Y", country: "IT", flag: "🇮🇹", maturityYears: 10 },
];

// --- Live yield fetch (cached) ---
interface CacheEntry {
  value: number | null;
  ts: number;
}
const yieldCache = new Map<string, CacheEntry>();
const CACHE_MS = 60_000;

async function fetchYahooYield(symbol: string): Promise<number | null> {
  const cached = yieldCache.get(symbol);
  if (cached && Date.now() - cached.ts < CACHE_MS) return cached.value;
  try {
    const res = await fetch(
      `https://query1.finance.yahoo.com/v8/finance/chart/${encodeURIComponent(symbol)}?interval=1d&range=1d`,
      {
        headers: { "User-Agent": "Mozilla/5.0 (compatible; dqi-immunization/1.0)" },
        signal: AbortSignal.timeout(4000),
      },
    );
    if (!res.ok) throw new Error(`yahoo ${res.status}`);
    const json = (await res.json()) as {
      chart?: { result?: Array<{ meta?: { regularMarketPrice?: number } }> };
    };
    const price = json.chart?.result?.[0]?.meta?.regularMarketPrice ?? null;
    yieldCache.set(symbol, { value: price, ts: Date.now() });
    return price;
  } catch {
    yieldCache.set(symbol, { value: cached?.value ?? null, ts: Date.now() });
    return cached?.value ?? null;
  }
}

export async function searchBonds(query: string): Promise<Bond[]> {
  const q = query.trim().toLowerCase();
  const matches = q
    ? CATALOG.filter(
        (b) =>
          b.label.toLowerCase().includes(q) ||
          b.country.toLowerCase().includes(q) ||
          b.id.includes(q),
      )
    : CATALOG;

  // Enrich with live yields in parallel (deduped by symbol via cache).
  return Promise.all(
    matches.map(async (b) => ({
      ...b,
      liveYield: b.yahooSymbol ? await fetchYahooYield(b.yahooSymbol) : null,
    })),
  );
}
