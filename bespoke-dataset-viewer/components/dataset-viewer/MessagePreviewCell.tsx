"use client"

import { TableCell } from "@/components/ui/table"
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip"

interface MessagePreviewCellProps {
  content: string
  column: string
}

export function MessagePreviewCell({ content, column }: MessagePreviewCellProps) {
  const shouldShowTooltip = column === "User Message" || column === "Assistant Message"

  if (!shouldShowTooltip) {
    return (
      <TableCell className="max-w-[400px] truncate">
        {content}
      </TableCell>
    )
  }

  return (
    <TableCell className="max-w-[400px] truncate">
      <TooltipProvider>
        <Tooltip delayDuration={300}>
          <TooltipTrigger asChild>
            <span className="cursor-pointer hover:text-blue-500 transition-colors">
              {content}
            </span>
          </TooltipTrigger>
          <TooltipContent side="bottom" className="max-w-[600px] whitespace-pre-wrap p-4">
            {content}
          </TooltipContent>
        </Tooltip>
      </TooltipProvider>
    </TableCell>
  )
}
