"use client";

import { useMemo, useState } from "react";

import { api, type RouterOutputs } from "~/trpc/react";

type Bond = RouterOutputs["bonds"]["search"][number];
type Size = 7 | 15 | 31;
const SIZES: Size[] = [7, 15, 31];

const fmtMat = (y: number) =>
  y < 1 ? `${Math.round(y * 12)}M` : `${y % 1 === 0 ? y : y.toFixed(1)}Y`;

export function ImmunizationApp() {
  return (
    <main className="relative min-h-screen">
      <div className="mx-auto max-w-3xl px-6 pt-16 pb-24 sm:px-8">
        <Masthead />

        <Section numeral="I" title="A liability you want to neutralize">
          <p>
            A pension fund owes a fixed sum years from now. It holds bonds to
            cover it. In between, interest rates move, and every move revalues
            both the obligation and the bonds meant to meet it. The actuary{" "}
            <Cite href="https://en.wikipedia.org/wiki/Immunization_(finance)">
              Frank Redington
            </Cite>{" "}
            asked, in 1952, whether a portfolio could be arranged so that the two
            move <em>in lockstep</em>: whatever rates do, the assets gain exactly
            what the liability gains. He called such a portfolio{" "}
            <strong className="font-semibold text-ink">immunized</strong>.
          </p>
          <p>
            The recipe is a hierarchy of matches. Match the present value. Then
            match its first derivative in yield, the{" "}
            <Cite href="https://en.wikipedia.org/wiki/Bond_duration">
              duration
            </Cite>
            . Then the second, the convexity. Each further match cancels another
            term in the Taylor expansion of the rate shock, and the immunization
            holds against ever-larger disturbances.
          </p>
        </Section>

        <Section numeral="II" title="Immunization is secretly an error-correcting code">
          <p>
            Here is the quiet fact this whole demonstration turns on. A bond pays
            cash at maturities{" "}
            <Mono>
              t<sub>1</sub>, t<sub>2</sub>, …
            </Mono>{" "}
            with weights{" "}
            <Mono>
              w<sub>i</sub>
            </Mono>
            . Duration, convexity, and their successors are nothing but the
            successive <em>moments</em> of that schedule,{" "}
            <Mono>
              Σ w<sub>i</sub> t<sub>i</sub>
              <sup>k</sup>
            </Mono>
            . To immunize through order <Mono>d</Mono> is to force the first{" "}
            <Mono>d</Mono> moments to agree with the liability, a system whose
            matrix is a{" "}
            <Cite href="https://en.wikipedia.org/wiki/Vandermonde_matrix">
              Vandermonde
            </Cite>{" "}
            of the maturities.
          </p>
          <p>
            But a Vandermonde / moment constraint, read over a finite field{" "}
            <Mono>
              GF(2<sup>m</sup>)
            </Mono>{" "}
            rather than the reals, is <em>precisely</em> the
            parity check of a{" "}
            <Cite href="https://en.wikipedia.org/wiki/BCH_code">BCH</Cite> /{" "}
            <Cite href="https://en.wikipedia.org/wiki/Reed%E2%80%93Solomon_error_correction">
              Reed–Solomon
            </Cite>{" "}
            code. The question <em>“is this portfolio immunized?”</em> becomes{" "}
            <em>“is this a valid codeword?”</em>. A single mis-tuned bond is an{" "}
            <strong className="font-semibold text-ink">error</strong> at one
            position; the moments it spoils are its{" "}
            <strong className="font-semibold text-ink">syndrome</strong>. Finding
            the odd-one-out that breaks immunization is exactly{" "}
            <em>decoding</em>.
          </p>
          <p>
            That last step deserves a caveat, because it is where the analogy is
            still doing real work rather than settled mathematics. Genuine
            immunization lives over the <em>reals</em>: durations are continuous,
            weights are signed and fractional, and “close enough” is a metric
            statement. Reading the moment constraint over{" "}
            <Mono>
              GF(2<sup>m</sup>)
            </Mono>{" "}
            buys the clean code structure but discards exactly that metric. The
            characteristic-two reduction here is a faithful{" "}
            <em>surrogate</em>, not a proven equivalence to the real problem, and
            mapping continuous duration matching onto a finite field is{" "}
            <strong className="font-semibold text-ink">not solved</strong>. A
            prime field{" "}
            <Mono>
              GF(<em>p</em>)
            </Mono>{" "}
            keeps more arithmetic structure and a fixed-point or lattice encoding
            may preserve the metric outright. Which encoding, if any, makes
            real-valued immunization decodable on hardware is open, and worth
            exploring.
          </p>
        </Section>

        <Section numeral="III" title="Decoded Quantum Interferometry">
          <p>
            In 2024 Jordan, Shutty, Wootters, Zalcman, Schmidhuber, King, Isakov
            and Babbush introduced{" "}
            <Cite href="https://arxiv.org/abs/2408.08292">
              Decoded Quantum Interferometry
            </Cite>{" "}
            (DQI), a way of turning a hard optimization into a{" "}
            <em>decoding</em> problem and then letting quantum interference do the
            decoding. One prepares a superposition weighted by a carefully chosen
            profile, applies the code’s parity map, and runs the classical
            decoder (embedded in the quantum circuit) <em>in reverse</em>. Amplitudes that correspond to good
            solutions interfere constructively; the rest cancel.
          </p>
          <p>
            Because immunization <em>is</em> a Reed–Solomon parity check, its
            decoder is the classic{" "}
            <Cite href="https://en.wikipedia.org/wiki/Berlekamp%E2%80%93Massey_algorithm">
              Berlekamp–Massey
            </Cite>{" "}
            algorithm. Run as a reversible quantum circuit, it concentrates
            amplitude on the syndrome that names the bond breaking the ladder.
            The instrument below builds that circuit from a ladder you choose and
            runs it. First on a simulator, then, if you like, on a real{" "}
            <strong className="font-semibold text-ink">IBM</strong> quantum
            processor.
          </p>
        </Section>

        <div className="mt-16 mb-10 rule" />

        <SectionHead
          numeral="IV"
          title="The instrument"
          dek="Compose a maturity ladder. One bond is secretly mis-tuned — watch the decoder pick out the odd-one-out, in simulation and on hardware."
        />
        <Instrument />

        <Section numeral="V" title="An honest accounting of scope">
          <p>
            What runs on the hardware is the validated{" "}
            <Mono>t = 1</Mono> syndrome-basis collapse. Three, four, or five
            qubits for seven, fifteen, or thirty-one bonds, small enough to stay
            beneath the device’s coherence wall, where it amplifies the
            immunizing readout several-fold over chance. It solves the{" "}
            <Mono>GF(2)</Mono> combinatorial surrogate of immunization (
            <em>which weight-one bond is the odd-one-out for the binarised moment
            syndrome</em>), faithful to immunization’s algebra but not yet to
            real-valued duration matching. Live yields are drawn from Yahoo
            Finance.
          </p>
        </Section>

        <References />
      </div>
    </main>
  );
}

/* ── The interactive instrument ─────────────────────────────────────── */

function Instrument() {
  const [size, setSize] = useState<Size>(7);
  const [query, setQuery] = useState("");
  const [selected, setSelected] = useState<Bond[]>([]);

  const search = api.bonds.search.useQuery(
    { query },
    { staleTime: 30_000, refetchOnWindowFocus: false },
  );
  const budget = api.quantum.budget.useQuery(undefined, {
    refetchInterval: 5000,
  });

  // Ladder = selected bonds sorted ascending by maturity (canonical order the
  // Python side also uses; the hidden saboteur is seeded by this ladder).
  const ladder = useMemo(
    () => [...selected].sort((a, b) => a.maturityYears - b.maturityYears),
    [selected],
  );
  const maturities = ladder.map((b) => b.maturityYears);
  const ready = selected.length === size;

  const preview = api.immunization.preview.useMutation();
  const run = api.quantum.run.useMutation();

  const [jobId, setJobId] = useState<string | null>(null);
  const poll = api.quantum.poll.useQuery(
    { jobId: jobId!, size, maturities },
    {
      enabled: !!jobId,
      refetchInterval: (q) => (q.state.data?.done ? false : 4000),
      refetchOnWindowFocus: false,
    },
  );

  // The bond(s) the decoder revealed as mis-tuned, from whichever result is
  // showing (hardware takes precedence once done). Used to light up the ladder.
  const revealed = useMemo(() => {
    const shown = poll.data?.done ? poll.data : preview.data;
    return new Set(shown?.decoded.bond_details.map((d) => d.bond) ?? []);
  }, [poll.data, preview.data]);

  // The hidden saboteur is seeded by the ladder, so any change to it invalidates
  // a shown result — clear it so the reveal never goes stale.
  function resetResults() {
    setJobId(null);
    preview.reset();
  }
  function addBond(b: Bond) {
    if (selected.length >= size) return;
    setSelected((s) => [...s, b]);
    resetResults();
  }
  function removeBond(i: number) {
    setSelected((s) => s.filter((_, idx) => idx !== i));
    resetResults();
  }
  function fillSample() {
    // Deterministic sample ladder of the right length from the catalog.
    const pool = search.data ?? [];
    const us = pool.filter((b) => b.country === "US");
    const chosen: Bond[] = [];
    for (let i = 0; i < size && i < us.length * 4; i++) {
      chosen.push(us[i % us.length]!);
    }
    setSelected(chosen.slice(0, size));
    resetResults();
  }

  async function doSimulate() {
    setJobId(null);
    await preview.mutateAsync({ size, maturities });
  }
  async function doRun() {
    setJobId(null);
    const res = await run.mutateAsync({ size, maturities });
    setJobId(res.job_id);
    void budget.refetch();
  }

  return (
    <div className="grid gap-8 lg:grid-cols-[1fr_1.05fr]">
      {/* LEFT: configure the portfolio */}
      <section className="space-y-7">
        <Card numeral="i" title="Portfolio size">
          <div className="grid grid-cols-3 gap-px bg-rule">
            {SIZES.map((s) => (
              <button
                key={s}
                onClick={() => {
                  setSize(s);
                  setSelected([]);
                  resetResults();
                }}
                className={`px-3 py-3 text-center transition ${
                  size === s
                    ? "bg-ink text-paper"
                    : "bg-panel text-ink-soft hover:bg-paper-2"
                }`}
              >
                <span className="block font-display text-xl font-semibold leading-none">
                  {s}
                </span>
                <span className="mt-1 block font-mono text-[9px] uppercase tracking-[0.12em] opacity-70">
                  GF(2^{Math.log2(s + 1)}) · {Math.log2(s + 1)} qubits
                </span>
              </button>
            ))}
          </div>
        </Card>

        <Card
          numeral="ii"
          title={`Pick bonds · ${selected.length}/${size}`}
          right={
            <button
              onClick={fillSample}
              className="font-mono text-[10px] uppercase tracking-[0.12em] text-accent underline-offset-4 hover:underline"
            >
              auto-fill
            </button>
          }
        >
          <input
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Search treasuries & sovereigns. 10Y, US, Bund…"
            className="w-full border border-rule bg-paper px-3 py-2 text-sm text-ink outline-none placeholder:text-ink-faint focus:border-ink"
          />
          <div className="mt-3 h-52 divide-y divide-rule/60 overflow-y-auto border border-rule/60">
            {(search.data ?? []).map((b) => (
              <button
                key={b.id}
                onClick={() => addBond(b)}
                disabled={selected.length >= size}
                className="flex w-full items-center justify-between bg-panel px-3 py-2 text-left text-sm transition hover:bg-paper-2 disabled:cursor-not-allowed disabled:opacity-40"
              >
                <span className="text-ink">
                  {b.flag} {b.label}
                </span>
                <span className="font-mono text-xs text-ink-faint tnum">
                  {fmtMat(b.maturityYears)}
                  {b.liveYield != null && (
                    <span className="ml-2 text-accent">
                      {b.liveYield.toFixed(2)}
                    </span>
                  )}
                </span>
              </button>
            ))}
            {search.isLoading && (
              <p className="px-3 py-2 font-mono text-[11px] text-ink-faint">
                searching…
              </p>
            )}
          </div>
        </Card>

        <Card numeral="iii" title="The maturity ladder">
          <p className="mb-3 text-sm leading-relaxed text-ink-soft">
            Sorted by maturity, then mapped onto the code’s locator set α
            <sup className="text-[0.8em]">i</sup>. One bond is secretly mis-tuned
            — the odd-one-out that breaks immunization. You don’t pick it; it’s
            seeded by the ladder. Simulate or run on hardware, and the decoder
            reveals which one.
          </p>
          {ladder.length === 0 ? (
            <div className="flex h-52 items-center justify-center border border-dashed border-rule/70 bg-paper/40 text-center font-mono text-[11px] uppercase tracking-[0.12em] text-ink-faint">
              no bonds yet
            </div>
          ) : (
            <div className="h-52 divide-y divide-rule/60 overflow-y-auto border border-rule/60">
              {ladder.map((b, i) => {
                const isSaboteur = revealed.has(i);
                return (
                  <div
                    key={`${b.id}-${i}`}
                    className={`flex items-center justify-between px-3 py-2 text-sm transition ${
                      isSaboteur ? "bg-accent/20" : "bg-panel"
                    }`}
                  >
                    <div className="flex items-center gap-2.5">
                      <span className="w-7 font-mono text-[11px] text-ink-faint">
                        α<sup className="text-[0.8em]">{i}</sup>
                      </span>
                      <span className="text-ink">
                        {b.flag} {b.label}
                      </span>
                      {isSaboteur && (
                        <span className="font-mono text-[9px] font-semibold uppercase tracking-[0.1em] text-accent">
                          ⚠ mis-tuned
                        </span>
                      )}
                    </div>
                    <div className="flex items-center gap-3 font-mono text-xs text-ink-faint tnum">
                      <span>{fmtMat(b.maturityYears)}</span>
                      <button
                        onClick={() => removeBond(i)}
                        className="text-ink-faint transition hover:text-accent"
                        aria-label="remove bond"
                      >
                        ✕
                      </button>
                    </div>
                  </div>
                );
              })}
            </div>
          )}
        </Card>

        <div className="grid grid-cols-2 gap-3">
          <button
            onClick={doSimulate}
            disabled={!ready || preview.isPending}
            className="border border-ink bg-transparent px-4 py-3 font-display text-base font-semibold text-ink transition hover:bg-ink hover:text-paper disabled:cursor-not-allowed disabled:opacity-30 disabled:hover:bg-transparent disabled:hover:text-ink"
          >
            {preview.isPending ? "Simulating…" : "Simulate · free"}
          </button>
          <button
            onClick={doRun}
            disabled={!ready || run.isPending || poll.data?.done === false}
            className="border border-accent bg-accent px-4 py-3 font-display text-base font-semibold text-paper transition hover:bg-accent-soft disabled:cursor-not-allowed disabled:opacity-30"
          >
            {run.isPending ? "Submitting…" : "Run on IBM QPU"}
          </button>
        </div>
        {!ready && (
          <p className="text-center font-mono text-[11px] uppercase tracking-[0.12em] text-ink-faint">
            {`pick ${size - selected.length} more bond${size - selected.length === 1 ? "" : "s"} to enable`}
          </p>
        )}
        {run.error && (
          <p className="font-mono text-xs text-accent">{run.error.message}</p>
        )}
        {budget.data && (
          <p className="text-center font-mono text-[10px] uppercase tracking-[0.1em] text-ink-faint tnum">
            QPU budget · window {budget.data.windowUsed}/{budget.data.windowMax}{" "}
            · in-flight {budget.data.inflight}/{budget.data.inflightMax} · total{" "}
            {budget.data.totalRuns}/{budget.data.totalMax}
          </p>
        )}
      </section>

      {/* RIGHT: results */}
      <section className="space-y-7">
        {preview.data && (
          <ResultCard
            heading="Simulator result"
            tone="slate"
            data={preview.data}
          />
        )}
        {jobId && (
          <JobStatus
            jobId={jobId}
            status={poll.data?.status ?? "SUBMITTING"}
            done={poll.data?.done ?? false}
            error={poll.error?.message}
          />
        )}
        {poll.data?.done && (
          <ResultCard
            heading="IBM hardware result"
            tone="accent"
            data={poll.data}
          />
        )}
        {!preview.data && !jobId && <EmptyState />}
      </section>
    </div>
  );
}

/* ── Essay furniture ────────────────────────────────────────────────── */

function Masthead() {
  return (
    <header className="text-center">
      <p className="font-mono text-[11px] uppercase tracking-[0.32em] text-ink-faint">
        MMXXVI
      </p>
      <h1 className="mt-5 font-display text-[2.6rem] font-semibold leading-[1.08] tracking-tight text-ink sm:text-[3.4rem]">
        On the Immunization of Bond Portfolios
        <span className="block font-normal italic text-accent">
          by DQI
        </span>
      </h1>
      <p className="mx-auto mt-6 max-w-xl text-lg leading-relaxed text-ink-soft">
        A portfolio that cannot be affected by interest rates is an
        error-correcting code in disguise, and a quantum computer can decode it.
      </p>
      <div className="mx-auto mt-9 h-px w-24 bg-accent/60" />
    </header>
  );
}

function Section({
  numeral,
  title,
  children,
}: {
  numeral: string;
  title: string;
  children: React.ReactNode;
}) {
  return (
    <section className="mt-16">
      <SectionHead numeral={numeral} title={title} />
      <div className="space-y-5 text-[1.08rem] leading-[1.78] text-ink-soft">
        {children}
      </div>
    </section>
  );
}

function SectionHead({
  numeral,
  title,
  dek,
}: {
  numeral: string;
  title: string;
  dek?: string;
}) {
  return (
    <div className="mb-5">
      <div className="flex items-baseline gap-4">
        <span className="font-display text-2xl font-medium leading-none text-accent">
          {numeral}
        </span>
        <h2 className="font-display text-[1.7rem] font-semibold leading-tight tracking-tight text-ink">
          {title}
        </h2>
      </div>
      {dek && (
        <p className="mt-3 max-w-2xl text-base italic leading-relaxed text-ink-faint">
          {dek}
        </p>
      )}
    </div>
  );
}

function Cite({ href, children }: { href: string; children: React.ReactNode }) {
  return (
    <a
      href={href}
      target="_blank"
      rel="noreferrer"
      className="text-accent underline decoration-accent/40 decoration-1 underline-offset-[3px] transition hover:decoration-accent"
    >
      {children}
    </a>
  );
}

function Mono({ children }: { children: React.ReactNode }) {
  return (
    <span className="whitespace-nowrap font-mono text-[0.85em] text-ink tnum [&_sub]:text-[0.7em] [&_sup]:text-[0.7em]">
      {children}
    </span>
  );
}

function References() {
  const refs = [
    {
      n: "1",
      cite: "S. P. Jordan, N. Shutty, M. Wootters, A. Zalcman, A. Schmidhuber, R. King, S. V. Isakov, R. Babbush. “Optimization by Decoded Quantum Interferometry.”",
      href: "https://arxiv.org/abs/2408.08292",
      label: "arXiv:2408.08292",
    },
  ];
  return (
    <section className="mt-16">
      <div className="mb-6 rule" />
      <h2 className="mb-5 font-mono text-[11px] uppercase tracking-[0.2em] text-ink-faint">
        References & further reading
      </h2>
      <ol className="space-y-3">
        {refs.map((r) => (
          <li key={r.n} className="flex gap-3 text-sm leading-relaxed">
            <span className="font-display text-base font-semibold text-accent">
              {r.n}
            </span>
            <span className="text-ink-soft">
              {r.cite}{" "}
              <a
                href={r.href}
                target="_blank"
                rel="noreferrer"
                className="whitespace-nowrap font-mono text-[11px] text-accent underline underline-offset-2"
              >
                {r.label} ↗
              </a>
            </span>
          </li>
        ))}
      </ol>
    </section>
  );
}

/* ── Instrument furniture ───────────────────────────────────────────── */

function Card({
  numeral,
  title,
  right,
  children,
}: {
  numeral: string;
  title: string;
  right?: React.ReactNode;
  children: React.ReactNode;
}) {
  return (
    <div className="border border-rule bg-panel/70 p-5">
      <div className="mb-4 flex items-center justify-between border-b border-rule/70 pb-2.5">
        <h3 className="flex items-baseline gap-2 font-display text-lg font-semibold tracking-tight text-ink">
          <span className="font-mono text-[11px] uppercase tracking-[0.15em] text-accent">
            {numeral}
          </span>
          {title}
        </h3>
        {right}
      </div>
      {children}
    </div>
  );
}

function Stat({
  label,
  value,
  sub,
}: {
  label: string;
  value: string;
  sub?: string;
}) {
  return (
    <div className="border border-rule/70 bg-paper/50 p-3">
      <div className="font-mono text-[9px] uppercase tracking-[0.12em] text-ink-faint">
        {label}
      </div>
      <div className="mt-1 font-display text-2xl font-semibold leading-none text-ink tnum">
        {value}
      </div>
      {sub && (
        <div className="mt-1 font-mono text-[10px] text-ink-faint tnum">
          {sub}
        </div>
      )}
    </div>
  );
}

type ScoreData = RouterOutputs["immunization"]["preview"];

function ResultCard({
  heading,
  tone,
  data,
}: {
  heading: string;
  tone: "slate" | "accent";
  data: ScoreData;
}) {
  const accentBar = tone === "accent" ? "bg-accent" : "bg-slate";
  const accentText = tone === "accent" ? "text-accent" : "text-slate";
  const dist = data.solution_dist;
  const max = Math.max(...Object.values(dist), 0.0001);
  const optBits = data.optimum;
  return (
    <div className="border border-rule bg-panel/70 p-5">
      <h3 className="mb-4 flex items-center gap-2 border-b border-rule/70 pb-2.5 font-display text-lg font-semibold tracking-tight text-ink">
        <span className={`h-2 w-2 ${accentBar}`} />
        {heading}
      </h3>
      <div className="grid grid-cols-3 gap-2">
        <Stat
          label="Mean satisfied"
          value={`${data.mean.toFixed(2)}/${data.m}`}
          sub={`random ${data.random_mean}`}
        />
        <Stat
          label={`P(opt=${optBits})`}
          value={`${(data.p_opt * 100).toFixed(1)}%`}
          sub={`random ${(data.random_p * 100).toFixed(1)}%`}
        />
        <Stat
          label="Amplification"
          value={`${data.lift.toFixed(1)}×`}
          sub="vs random"
        />
      </div>

      <div className="mt-5">
        <div className="mb-2 font-mono text-[10px] uppercase tracking-[0.12em] text-ink-faint">
          Solution distribution · {data.shots.toLocaleString()} shots
        </div>
        <div className="space-y-1.5">
          {Object.entries(dist)
            .sort((a, b) => Number(b[1]) - Number(a[1]))
            .slice(0, 8)
            .map(([sol, p]) => (
              <div key={sol} className="flex items-center gap-2 text-xs">
                <span className="w-10 font-mono text-ink-faint">
                  {Number(sol).toString(2).padStart(data.n_syn, "0")}
                </span>
                <div className="h-3 flex-1 overflow-hidden bg-paper-2">
                  <div
                    className={`h-full ${accentBar}`}
                    style={{ width: `${(Number(p) / max) * 100}%` }}
                  />
                </div>
                <span className="w-12 text-right font-mono text-ink-faint tnum">
                  {(Number(p) * 100).toFixed(1)}%
                </span>
              </div>
            ))}
        </div>
      </div>

      <div className="mt-5 border border-rule/70 bg-paper/50 p-3 text-sm">
        <div className="mb-1.5 flex items-center justify-between font-mono text-[10px] uppercase tracking-[0.12em] text-ink-faint">
          <span>Decoded immunizing readout</span>
          {data.decoded.bond_details.length > 0 &&
            (data.recovered ? (
              <span className={`font-semibold ${accentText}`}>
                ✓ found the hidden bond
              </span>
            ) : (
              <span className="font-semibold text-ink-faint">
                ✗ noise overwhelmed the readout
              </span>
            ))}
        </div>
        {data.decoded.bond_details.length > 0 ? (
          data.decoded.bond_details.map((d) => (
            <div key={d.bond} className="leading-relaxed text-ink-soft">
              hidden mis-tuned bond{" "}
              <span className={`font-semibold ${accentText}`}>#{d.bond}</span> ·
              maturity {d.maturity.toFixed(2)}y · locator α
              <sup className="text-[0.8em]">{d.bond}</sup> ={" "}
              <span className="font-mono">{d.locator}</span>
            </div>
          ))
        ) : (
          <div className="text-ink-faint">
            no single odd-one-out. Syndrome consistent with full match.
          </div>
        )}
      </div>
    </div>
  );
}

function JobStatus({
  jobId,
  status,
  done,
  error,
}: {
  jobId: string;
  status: string;
  done: boolean;
  error?: string;
}) {
  return (
    <div className="border border-accent/50 bg-accent/[0.06] p-5">
      <div className="flex items-center gap-3">
        {!done && (
          <span className="relative flex h-2.5 w-2.5">
            <span className="absolute inline-flex h-full w-full animate-ping bg-accent opacity-60" />
            <span className="relative inline-flex h-2.5 w-2.5 bg-accent" />
          </span>
        )}
        <div>
          <div className="font-display text-base font-semibold text-ink">
            {done ? "Job complete" : "Running on IBM Quantum…"}
          </div>
          <div className="font-mono text-[11px] text-ink-faint">
            {jobId} · {status.replace("JobStatus.", "")}
          </div>
        </div>
      </div>
      {error && <p className="mt-2 font-mono text-xs text-accent">{error}</p>}
    </div>
  );
}

function EmptyState() {
  return (
    <div className="flex h-full min-h-72 items-center justify-center border border-dashed border-rule p-8 text-center">
      <div className="max-w-xs">
        <div className="font-display text-5xl text-accent/70">⌘</div>
        <p className="mt-4 font-display text-lg italic text-ink-soft">
          Awaiting instructions.
        </p>
        <p className="mt-1 text-sm text-ink-faint">
          Compose a ladder at left, then simulate or run it on the QPU. The
          result will appear here.
        </p>
      </div>
    </div>
  );
}
