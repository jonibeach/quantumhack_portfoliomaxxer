import "~/styles/globals.css";

import { type Metadata } from "next";
import { Cormorant_Garamond, EB_Garamond, IBM_Plex_Mono } from "next/font/google";

import { TRPCReactProvider } from "~/trpc/react";

export const metadata: Metadata = {
  title: "On the Immunization of Bond Portfolios by Quantum Interferometry",
  description:
    "Fixed-income immunization is a Reed–Solomon parity check, and Decoded Quantum Interferometry amplifies the immunizing portfolio on real IBM quantum hardware.",
  // Favicon comes from the file convention: src/app/icon.svg
};

// A reading-room palette: an old-style serif for prose, a high-contrast
// display serif for headings, and IBM's mono for the figures and numerals.
const ebGaramond = EB_Garamond({
  subsets: ["latin"],
  variable: "--font-eb",
});
const cormorant = Cormorant_Garamond({
  subsets: ["latin"],
  weight: ["400", "500", "600", "700"],
  variable: "--font-cormorant",
});
const plexMono = IBM_Plex_Mono({
  subsets: ["latin"],
  weight: ["400", "500"],
  variable: "--font-plex",
});

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html
      lang="en"
      className={`${ebGaramond.variable} ${cormorant.variable} ${plexMono.variable}`}
    >
      <body>
        <TRPCReactProvider>{children}</TRPCReactProvider>
      </body>
    </html>
  );
}
