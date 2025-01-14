"use client"

import { TableHead } from "@/components/ui/table"
import { ChevronDown, ChevronUp, GripVertical, Filter } from "lucide-react"
import { useSortable } from "@dnd-kit/sortable"
import { CSS } from "@dnd-kit/utilities"
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu"
import { Input } from "@/components/ui/input"
import { SortableHeaderProps } from "@/types/table"

export function SortableHeader({
  column,
  onSort,
  sortColumn,
  sortDirection,
  onFilter,
  filterValue,
}: SortableHeaderProps) {
  const { attributes, listeners, setNodeRef, transform, transition } = useSortable({
    id: column.key
  })

  const style = {
    transform: CSS.Transform.toString(transform),
    transition,
  }

  return (
    <TableHead ref={setNodeRef} style={style} className="whitespace-nowrap">
      <div className="flex items-center space-x-2">
        <div {...attributes} {...listeners}>
          <GripVertical className="h-4 w-4 cursor-grab" />
        </div>

        <span>{column.label}</span>

        <div className="flex items-center space-x-1" onClick={(e) => e.stopPropagation()}>
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <button
                className="hover:bg-muted rounded p-1"
                onClick={(e) => e.stopPropagation()}
              >
                <Filter className="h-4 w-4" />
              </button>
            </DropdownMenuTrigger>
            <DropdownMenuContent>
              <div className="p-2">
                <Input
                  placeholder={`Filter ${column.label}`}
                  value={filterValue}
                  onChange={(e) => onFilter(column.key, e.target.value)}
                  onClick={(e) => e.stopPropagation()}
                />
              </div>
            </DropdownMenuContent>
          </DropdownMenu>

          <button
            onClick={(e) => {
              e.stopPropagation()
              onSort(column.key)
            }}
            className="hover:bg-muted rounded p-1"
          >
            {sortColumn === column.key ? (
              sortDirection === "asc" ? (
                <ChevronUp className="h-4 w-4" />
              ) : (
                <ChevronDown className="h-4 w-4" />
              )
            ) : (
              <ChevronDown className="h-4 w-4 opacity-30" />
            )}
          </button>
        </div>
      </div>
    </TableHead>
  )
}
