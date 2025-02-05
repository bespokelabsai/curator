import type { Metadata } from "next";
import "./globals.css";
import { Toaster } from "@/components/ui/toaster"

export const metadata: Metadata = {
  title: "Curator Viewer",
  description: "A powerful dataset viewer and analysis tool",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
      <html lang="en" suppressHydrationWarning>
          <body suppressHydrationWarning>
        {children}
        <Toaster />
      </body>
    </html>
  )
}
