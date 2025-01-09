"use client"

import { Header } from "@/components/layout/Header"
import { Button } from "@/components/ui/button"
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu"
import { SortableTable } from "@/components/ui/sortable-table"
import { useToast } from "@/components/ui/use-toast"
import { cn, getColumnValue } from "@/lib/utils"
import { DataItem } from "@/types/dataset"
import { Column } from "@/types/table"
import { AnimatePresence } from "framer-motion"
import { FileText, Loader2, RefreshCcw } from "lucide-react"
import { useCallback, useEffect, useState } from "react"
import { DetailsSidebar } from "./DetailsSidebar"
import { DistributionChart } from "./DistributionChart"
import { TimeSeriesChart } from './TimeSeriesChart'

const COLUMNS: Column[] = [
  { key: "user_message", label: "User Message" },
  { key: "assistant_message", label: "Assistant Message" },
  { key: "prompt_tokens", label: "Prompt Tokens" },
  { key: "completion_tokens", label: "Completion Tokens" },
  { key: "generation_time", label: "Generation Time (s)" }
]

interface DatasetViewerProps {
  runHash?: string
  batchMode: boolean
}

function NoDataView({ batchMode, isPolling }: { batchMode: boolean, isPolling: boolean }) {
  return (
    <div className="flex flex-col items-center justify-center h-[60vh] p-8">
      <div className="bg-muted/30 rounded-full p-6 mb-6">
        <FileText className="w-12 h-12 text-muted-foreground" />
      </div>
      <h3 className="text-xl font-semibold mb-2">No responses available yet</h3>
      <p className="text-muted-foreground text-center max-w-md mb-4">
        {batchMode
          ? "Waiting for the first batch to complete. Once finished, responses will appear here in batches."
          : "Responses will appear here as they are generated. The table will update automatically. Check if the curator is still running."}
      </p>
      <div className="flex items-center gap-2 text-sm text-muted-foreground">
        {isPolling ? (
          <>
            <RefreshCcw className="w-4 h-4 animate-spin" />
            <span>Polling for new responses...</span>
          </>
        ) : (
          <span>Polling is paused</span>
        )}
      </div>
    </div>
  )
}

export function DatasetViewer({ runHash, batchMode }: DatasetViewerProps) {
  const [data, setData] = useState<DataItem[]>([])
  const [theme, setTheme] = useState<"light" | "dark">("light")
  const [mounted, setMounted] = useState(false)
  const [selectedDistribution, setSelectedDistribution] = useState<string | null>("requests")
  const [selectedItem, setSelectedItem] = useState<DataItem | null>(null)
  const { toast } = useToast()
  const [isPolling, setIsPolling] = useState(false)
  const [lastLineNumber, setLastLineNumber] = useState(0)
  const [isInitialLoad, setIsInitialLoad] = useState(true)
  const [newItemIds, setNewItemIds] = useState<Set<number>>(new Set())
  const [processedFiles, setProcessedFiles] = useState<string[]>([])

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

  const fetchNewResponses = useCallback(async () => {
    if (!runHash) return

    try {
      const queryParams = new URLSearchParams({
        batchMode: batchMode.toString(),
        ...(batchMode
          ? { processedFiles: processedFiles.join(',') }
          : { lastLine: lastLineNumber.toString() }
        )
      })

      const response = await fetch(`/api/responses/${runHash}?${queryParams}`)
      if (!response.ok) throw new Error('Failed to fetch responses')

      const responseData = await response.json()

      if (responseData.data && responseData.data.length > 0) {
        setNewItemIds(new Set(responseData.data.map((item: DataItem) => item.raw_response.id)))

        setData(prevData => [...responseData.data.reverse(), ...prevData])

        if (batchMode) {
          // Update processed files list
          setProcessedFiles(prev => [...prev, ...responseData.processedFiles])
        } else {
          // Update last line number for streaming mode
          setLastLineNumber(responseData.totalLines)
        }

        setTimeout(() => {
          setNewItemIds(new Set())
        }, 5000)

        toast({
          title: "New responses received",
          description: `${responseData.data.length} new responses added`,
        })
      }
    } catch (error) {
      console.error("Error fetching responses:", error)
      toast({
        variant: "destructive",
        title: "Error",
        description: "Failed to fetch new responses",
      })
    }
  }, [runHash, lastLineNumber, processedFiles, batchMode, toast])

  const handleInitialLoad = useCallback(async () => {
    if (!runHash) return

    try {
      const queryParams = new URLSearchParams({
        batchMode: batchMode.toString(),
        ...(batchMode
          ? { processedFiles: '' }
          : { lastLine: '0' }
        )
      })
      const response = await fetch(`/api/responses/${runHash}?${queryParams}`)
      if (!response.ok) throw new Error('Failed to fetch responses')
      const { data: initialData, totalLines, processedFiles: newProcessedFiles } = await response.json()

      if (initialData && Array.isArray(initialData)) {
        setData(initialData.reverse()) // Newest first

        if (batchMode && newProcessedFiles) {
          setProcessedFiles(newProcessedFiles)
        } else {
          setLastLineNumber(totalLines)
        }

        // Show initial data count
        toast({
          title: "Dataset loaded",
          description: `Loaded ${initialData.length} responses`,
        })
      }

      // Start polling after initial load
      setIsPolling(true)
    } catch (error) {
      console.error("Error reading responses file:", error)
      toast({
        variant: "destructive",
        title: "Error",
        description: "Failed to load dataset",
      })
    } finally {
      setIsInitialLoad(false)
    }
  }, [runHash, batchMode, toast])

  // Initial data load
  useEffect(() => {
    if (runHash && isInitialLoad) {
      handleInitialLoad()
    }
  }, [runHash, isInitialLoad, handleInitialLoad])

  // Modify the polling effect to avoid race conditions
  useEffect(() => {
    if (!isPolling || !runHash || isInitialLoad) return

    const pollInterval = setInterval(fetchNewResponses, 5000)

    return () => {
      clearInterval(pollInterval)
    }
  }, [isPolling, runHash, fetchNewResponses, isInitialLoad])

  // Don't render anything until mounted
  if (!mounted) {
    return null
  }

  return (
    <>
      <DetailsSidebar item={selectedItem} onClose={() => setSelectedItem(null)} />
      <div className="min-h-screen bg-background text-foreground overflow-x-hidden">
        <Header
          isLoading={isInitialLoad}
          isPolling={isPolling}
          onTogglePolling={() => setIsPolling(prev => !prev)}
          loadingText="Loading dataset..."
          pollingText="Polling for updates..."
        />

        <main className="container mx-auto p-4 overflow-x-hidden">
          <div className="mb-6 flex items-center justify-between">
            <div>
              <h2 className="text-2xl font-semibold text-foreground">Dataset Details</h2>
              <p className="text-sm text-muted-foreground">View and analyze your dataset responses</p>
            </div>
            {data.length > 0 && (
              <div className="flex gap-2">
                <DropdownMenu>
                  <DropdownMenuTrigger asChild>
                    <Button variant="outline">
                      {selectedDistribution
                        ? selectedDistribution === "requests"
                          ? "Requests & Responses"
                          : selectedDistribution.split('_').map(word =>
                              word.charAt(0).toUpperCase() + word.slice(1)
                            ).join(' ')
                        : 'Select Metric'}
                    </Button>
                  </DropdownMenuTrigger>
                  <DropdownMenuContent align="end">
                    <DropdownMenuItem onClick={() => setSelectedDistribution(null)}>
                      None
                    </DropdownMenuItem>
                    <DropdownMenuItem onClick={() => setSelectedDistribution("requests")}>
                      Requests & Responses
                    </DropdownMenuItem>
                    {["total_tokens", "prompt_tokens", "completion_tokens", "generation_time"].map((column) => (
                      <DropdownMenuItem key={column} onClick={() => setSelectedDistribution(column)}>
                        {column === "total_tokens" ? "Total Tokens" :
                          column === "prompt_tokens" ? "Prompt Tokens" :
                            column === "completion_tokens" ? "Completion Tokens" :
                              "Generation Time (s)"}
                      </DropdownMenuItem>
                    ))}
                  </DropdownMenuContent>
                </DropdownMenu>
              </div>
            )}
          </div>

          {isInitialLoad ? (
            <div className="flex items-center justify-center h-64">
              <div className="flex flex-col items-center gap-4">
                <Loader2 className="h-8 w-8 animate-spin" />
                <p className="text-muted-foreground">Loading dataset...</p>
              </div>
            </div>
          ) : data.length === 0 ? (
            <NoDataView batchMode={batchMode} isPolling={isPolling} />
          ) : (
            <>
              <div className="mb-8 space-y-4">
                {selectedDistribution && (
                  <div className="rounded-lg border bg-card p-4">
                    {selectedDistribution === "requests" ? (
                      <TimeSeriesChart data={data} />
                    ) : (
                      <DistributionChart
                        data={data}
                        column={selectedDistribution}
                      />
                    )}
                  </div>
                )}
              </div>

              <div className="rounded-lg border bg-card">
                <AnimatePresence>
                  <SortableTable
                    columns={COLUMNS}
                    data={data}
                    getRowKey={(item) => item.raw_response.id}
                    getCellContent={(item, columnKey) => getColumnValue(item, columnKey)}
                    onRowClick={(item) => setSelectedItem(item)}
                    truncateConfig={{
                      enabled: true,
                      maxLength: 150
                    }}
                    pageSize={10}
                    rowProps={(item) => ({
                      className: cn(
                        newItemIds.has(item.raw_response.id) && "bg-success/30 animate-highlight",
                        "transition-colors duration-300"
                      ),
                      layout: true,
                      initial: { opacity: 0, y: -20 },
                      animate: {
                        opacity: 1,
                        y: 0,
                        transition: {
                          duration: 0.2
                        }
                      },
                      exit: {
                        opacity: 0,
                        y: -20,
                        transition: {
                          duration: 0.2
                        }
                      },
                    })}
                  />
                </AnimatePresence>
              </div>
            </>
          )}
        </main>
      </div>
    </>
  )
}
