"use client"

import { useEffect, useState, useCallback } from "react"
import { useRouter } from "next/navigation"
import { SortableTable } from "@/components/ui/sortable-table"
import { Column } from "@/types/table"
import { Run } from "@/types/dataset"
import { useToast } from "@/components/ui/use-toast"
import { Loader2 } from "lucide-react"
import { cn } from "@/lib/utils"
import { Header } from "@/components/layout/Header"
import { AlertCircle } from "lucide-react"
import { PythonHighlighter } from "@/components/ui/python-highlighter"

const COLUMNS: Column[] = [
  { key: "created_time", label: "Created" },
  { key: "last_edited_time", label: "Last Updated" },
  { key: "model_name", label: "Model" },
  { key: "prompt_func", label: "Prompter Function" },
  { key: "response_format", label: "Response Format" }
]

const EXAMPLE_CODE = `from typing import Dict, List

from datasets import Dataset
from pydantic import BaseModel, Field

from bespokelabs import curator


class Poem(BaseModel):
    poem: str = Field(description="A poem.")


class Poems(BaseModel):
    poems: List[Poem] = Field(description="A list of poems.")


class Poet(curator.LLM):
    response_format = Poems

    def prompt(self, input: Dict) -> str:
        return f"Write two poems about {input['topic']}."

    def parse(self, input: Dict, response: Poems) -> Dict:
        return [{"topic": input["topic"], "poem": p.poem} for p in response.poems]


poet = Poet(model_name="gpt-4o-mini")

topics = Dataset.from_dict({"topic": ["Urban loneliness in a bustling city", "Beauty of Bespoke Labs's Curator library"]})
poem = poet(topics)
print(poem.to_pandas())
# Output:
#                                       topic                                               poem
# 0       Urban loneliness in a bustling city  In the city’s heart, where the lights never di...
# 1       Urban loneliness in a bustling city  Steps echo loudly, pavement slick with rain,\n...
# 2  Beauty of Bespoke Labs's Curator library  In the heart of Curation’s realm,  \nWhere art...
# 3  Beauty of Bespoke Labs's Curator library  Step within the library’s embrace,  \nA sanctu...
`

export function RunsTable() {
  const [runs, setRuns] = useState<Run[]>([])
  const [isLoading, setIsLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [isPolling, setIsPolling] = useState(true)
  const [lastCreatedTime, setLastCreatedTime] = useState<string | null>(null)
  const [newRunIds, setNewRunIds] = useState<Set<string>>(new Set())
  const router = useRouter()
  const { toast } = useToast()
  const [noCacheFound, setNoCacheFound] = useState<{ message: string; path: string } | null>(null)

  const fetchRuns = useCallback(async (isInitial = false) => {
    try {
      const queryParams = lastCreatedTime && !isInitial
        ? `?lastCreatedTime=${lastCreatedTime}`
        : ''

      const response = await fetch(`/api/runs${queryParams}`)
      const data = await response.json()

      if (response.status === 404 && data.error === 'NO_CACHE_DB') {
        setNoCacheFound({ message: data.message, path: data.path })
        setIsPolling(false) // Stop polling if no cache exists
        setIsLoading(false)
        return
      }

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`)
      }

      if (!Array.isArray(data)) {
        throw new Error('Data is not in the expected format')
      }

      if (data.length > 0) {
        if (isInitial) {
          setRuns(data)
          setLastCreatedTime(data[0].created_time)
        } else if (data.length > 0) {
          // Mark new runs for highlighting
          const newIds = new Set(data.map(run => run.run_hash))
          setNewRunIds(newIds)

          // Add new runs to the top
          setRuns(prevRuns => [...data, ...prevRuns])
          setLastCreatedTime(data[0].created_time)

          // Show toast for new runs
          toast({
            title: "New runs available",
            description: `${data.length} new runs added`,
          })

          // Clear highlighting after 5 seconds
          setTimeout(() => {
            setNewRunIds(new Set())
          }, 5000)
        }
      }
    } catch (error) {
      console.error('Failed to fetch runs:', error)
      setError(error instanceof Error ? error.message : 'Failed to fetch runs')
      toast({
        variant: "destructive",
        title: "Error",
        description: "Failed to fetch new runs",
      })
    } finally {
      setIsLoading(false)
    }
  }, [lastCreatedTime, toast])

  // Initial load
  useEffect(() => {
    fetchRuns(true)
  }, [fetchRuns])

  // Polling effect
  useEffect(() => {
    if (!isPolling) return

    const pollInterval = setInterval(() => {
      fetchRuns(false)
    }, 5000)

    return () => clearInterval(pollInterval)
  }, [isPolling, fetchRuns])

  const getCellContent = (run: Run, columnKey: string) => {
    switch (columnKey) {
      case "created_time":
        return new Date(run.created_time).toLocaleString()
      case "last_edited_time":
        return run.last_edited_time === '-'
          ? '-'
          : new Date(run.last_edited_time).toLocaleString()
      default:
        return run[columnKey as keyof Run]
    }
  }

  if (error) return <div>Error: {error}</div>
  return (
    <div className="min-h-screen bg-background text-foreground overflow-x-hidden">
      <Header
        isLoading={isLoading}
        isPolling={isPolling}
        onTogglePolling={() => setIsPolling(prev => !prev)}
        pollingText="Polling for updates..."
        loadingText="Loading runs..."
      />

      <main className="container mx-auto p-4">
        <div className="mb-6">
          <h2 className="text-2xl font-semibold text-foreground">Curator Runs History</h2>
          <p className="text-sm text-muted-foreground">View and analyze your past Bespoke Curator runs</p>
        </div>

        {noCacheFound ? (
          <div className="rounded-lg border border-yellow-200 bg-yellow-50 dark:border-yellow-900/50 dark:bg-yellow-900/20 p-4 my-4">
            <div className="flex flex-col space-y-4">
              <div className="flex items-start space-x-3">
                <AlertCircle className="h-5 w-5 text-yellow-600 dark:text-yellow-500 mt-0.5" />
                <div>
                  <h3 className="font-medium text-yellow-600 dark:text-yellow-500">
                    No Cache Database Found
                  </h3>
                  <p className="text-sm text-yellow-600 dark:text-yellow-400 mt-1">
                    {noCacheFound.message}
                  </p>
                  <p className="text-xs text-yellow-500 dark:text-yellow-400 mt-2 font-mono">
                    Expected location: {noCacheFound.path}
                  </p>
                </div>
              </div>

              <div className="mt-4 rounded-lg p-4">
                <p className="text-sm text-yellow-400 mb-2">Try running this example to create your first curator run:</p>
                <PythonHighlighter code={EXAMPLE_CODE} />
              </div>
            </div>
          </div>
        ) : isLoading ? (
          <div className="flex items-center justify-center h-[calc(100vh-200px)]">
            <div className="flex flex-col items-center gap-4">
              <Loader2 className="h-8 w-8 animate-spin" />
              <p className="text-muted-foreground">Loading runs...</p>
            </div>
          </div>
        ) : runs.length === 0 ? (
          <div className="flex items-center justify-center h-[calc(100vh-200px)]">
            <p className="text-muted-foreground">No runs found</p>
          </div>
        ) : (
          <div className="space-y-4">
            <SortableTable
              columns={COLUMNS}
              data={runs}
              getRowKey={(run) => run.id}
              getCellContent={getCellContent}
              onRowClick={(run) => router.push(`/dataset/${run.run_hash}?batchMode=${run.batch_mode}`)}
              truncateConfig={{ enabled: true, maxLength: 100 }}
              pageSize={10}
              rowProps={(run) => ({
                className: cn(
                  newRunIds.has(run.id) && "bg-success/30 animate-highlight",
                  "transition-colors duration-300"
                )
              })}
            />
          </div>
        )}
      </main>
    </div>
  )
}
