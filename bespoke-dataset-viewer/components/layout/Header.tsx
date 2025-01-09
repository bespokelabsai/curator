"use client"

import { Button } from "@/components/ui/button"
import { Moon, Sun, Loader2 } from "lucide-react"
import Image from "next/image"
import { useRouter } from "next/navigation"
import { useEffect, useState } from "react"

interface HeaderProps {
  isLoading?: boolean
  isPolling?: boolean
  onTogglePolling?: () => void
  showPolling?: boolean
  loadingText?: string
  pollingText?: string
}

export function Header({
  isLoading,
  isPolling,
  onTogglePolling,
  showPolling = true,
  loadingText = "Loading dataset...",
  pollingText = "Polling for updates..."
}: HeaderProps) {
  const router = useRouter()
  const [theme, setTheme] = useState<"light" | "dark">("light")
  const [mounted, setMounted] = useState(false)

  useEffect(() => {
    const systemPreference = window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light'
    const savedTheme = localStorage.getItem('theme') as 'light' | 'dark' | null
    setTheme(savedTheme || systemPreference)
    setMounted(true)
  }, [])

  useEffect(() => {
    if (!mounted) return

    if (theme === 'dark') {
      document.documentElement.classList.add('dark')
    } else {
      document.documentElement.classList.remove('dark')
    }
    localStorage.setItem('theme', theme)
  }, [theme, mounted])

  const toggleTheme = () => {
    setTheme(prev => prev === 'light' ? 'dark' : 'light')
  }

  if (!mounted) return null

  return (
    <header className="border-b sticky top-0 z-50 bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/75">
      <div className="container mx-auto px-4 py-4 flex items-center justify-between">
        <div className="flex items-center gap-2 cursor-pointer" onClick={() => router.push('/')}>
          <Image
            src="/Bespoke-Labs-Logomark-Red-on-Mint.svg"
            alt="Bespoke Logo"
            width={32}
            height={32}
            className="object-contain"
          />
          <h1 className="text-2xl font-bold">Bespoke Dataset Viewer</h1>
        </div>
        <div className="flex items-center gap-4">
          {isLoading ? (
            <div className="flex items-center gap-2 text-sm text-muted-foreground">
              <Loader2 className="h-4 w-4 animate-spin" />
              <span>{loadingText}</span>
            </div>
          ) : isPolling && showPolling ? (
            <div className="flex items-center gap-2 text-sm text-muted-foreground">
              <Loader2 className="h-4 w-4 animate-spin" />
              <span>{pollingText}</span>
            </div>
          ) : null}
          {showPolling && onTogglePolling && (
            <Button
              variant="outline"
              size="sm"
              onClick={onTogglePolling}
              disabled={isLoading}
            >
              {isPolling ? 'Stop' : 'Start'} Updates
            </Button>
          )}
          <Button variant="ghost" size="icon" onClick={toggleTheme}>
            {theme === 'light' ? (
              <Moon className="h-5 w-5" />
            ) : (
              <Sun className="h-5 w-5" />
            )}
            <span className="sr-only">Toggle theme</span>
          </Button>
        </div>
      </div>
    </header>
  )
}
