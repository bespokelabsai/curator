"use client"

import { Button } from "@/components/ui/button"
import { CardContent } from "@/components/ui/card"
import { Separator } from "@/components/ui/separator"
import { Copy } from "lucide-react"
import { DataItem } from "@/types/dataset"
import { useCallback } from "react"
import { Sheet, SheetContent } from "@/components/ui/sheet"

interface DetailsSidebarProps {
  item: DataItem | null
  onClose: () => void
}

export function DetailsSidebar({ item, onClose }: DetailsSidebarProps) {
  const copyToClipboard = useCallback(async (text: string) => {
    try {
      await navigator.clipboard.writeText(text)
      alert("Copied to clipboard!")
    } catch (err) {
      console.error("Failed to copy:", err)
      alert("Failed to copy to clipboard")
    }
  }, [])

  if (!item) return null

  return (
    <Sheet open={!!item} onOpenChange={() => item && onClose()}>
      <SheetContent 
        side="right" 
        className="w-full sm:w-[540px] p-0 fixed inset-y-0 border-l"
        style={{ height: '100vh' }}
      >
        <div className="h-full flex flex-col">
          <div className="p-6 border-b flex items-center justify-between">
            <h2 className="text-lg font-semibold">Response Details</h2>
          </div>
          
          <div className="flex-1 overflow-y-auto p-6">
            <CardContent className="space-y-6">
              <div className="space-y-2">
                <h3 className="text-lg font-semibold">Model</h3>
                <p className="text-sm text-muted-foreground">{item.request.model}</p>
              </div>
              <Separator />
              <div className="space-y-2">
                <h3 className="text-lg font-semibold">User Message</h3>
                <p className="text-sm text-muted-foreground whitespace-pre-wrap">
                  {item.request.messages.find(m => m.role === "user")?.content}
                </p>
                <Button 
                  onClick={() => copyToClipboard(item.request.messages.find(m => m.role === "user")?.content || "")} 
                  variant="outline" 
                  size="sm" 
                  className="mt-2"
                >
                  <Copy className="h-4 w-4 mr-2" />
                  Copy
                </Button>
              </div>
              <Separator />
              <div className="space-y-2">
                <h3 className="text-lg font-semibold">Assistant Message</h3>
                <p className="text-sm text-muted-foreground whitespace-pre-wrap">
                  {typeof item.response === 'object' 
                    ? JSON.stringify(item.response, null, 2)
                    : item.response}
                </p>
                <Button 
                  onClick={() => copyToClipboard(
                    typeof item.response === 'object' 
                      ? JSON.stringify(item.response, null, 2)
                      : item.response || ""
                  )} 
                  variant="outline" 
                  size="sm" 
                  className="mt-2"
                >
                  <Copy className="h-4 w-4 mr-2" />
                  Copy
                </Button>
              </div>
              <Separator />
              <div className="space-y-2">
                <h3 className="text-lg font-semibold">Token Usage</h3>
                <div className="grid grid-cols-3 gap-4">
                  <div className="space-y-1">
                    <p className="text-sm font-medium">Total</p>
                    <p className="text-2xl font-bold">{item.raw_response.usage.total_tokens}</p>
                  </div>
                  <div className="space-y-1">
                    <p className="text-sm font-medium">Prompt</p>
                    <p className="text-2xl font-bold">{item.raw_response.usage.prompt_tokens}</p>
                  </div>
                  <div className="space-y-1">
                    <p className="text-sm font-medium">Completion</p>
                    <p className="text-2xl font-bold">{item.raw_response.usage.completion_tokens}</p>
                  </div>
                </div>
              </div>
            </CardContent>
          </div>
        </div>
      </SheetContent>
    </Sheet>
  )
}