"use client"

import { useState, useMemo, useCallback } from 'react'
import { DndContext, DragEndEvent } from "@dnd-kit/core"
import { SortableContext, arrayMove } from "@dnd-kit/sortable"
import {
  Table,
  TableBody,
  TableCell,
  TableHeader,
  TableRow,
} from "@/components/ui/table"
import { SortableHeader } from "./sortable-header"
import { SortableTableProps, SortDirection } from "@/types/table"
import { Tooltip } from "@/components/ui/tooltip"
import { TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip"
import { motion } from "framer-motion"
import { cn } from "@/lib/utils"
import {
  Pagination,
  PaginationContent,
  PaginationItem,
  PaginationLink,
  PaginationNext,
  PaginationPrevious,
} from "@/components/ui/pagination"

export function SortableTable({
  columns,
  data,
  getRowKey,
  getCellContent,
  onRowClick,
  initialSortColumn = "",
  initialSortDirection = "asc",
  truncateConfig = { enabled: false, maxLength: 100 },
  rowProps,
  pageSize = 10,
}: SortableTableProps) {
  const [sortColumn, setSortColumn] = useState<string>(initialSortColumn)
  const [sortDirection, setSortDirection] = useState<SortDirection>(initialSortDirection)
  const [filters, setFilters] = useState<Record<string, string>>({})
  const [columnOrder, setColumnOrder] = useState(columns.map(col => col.key))
  const [currentPage, setCurrentPage] = useState(1)

  const handleSort = useCallback((column: string) => {
    if (sortColumn === column) {
      setSortDirection(prev => prev === "asc" ? "desc" : "asc")
    } else {
      setSortColumn(column)
      setSortDirection("asc")
    }
  }, [sortColumn])

  const handleFilter = useCallback((column: string, value: string) => {
    setFilters(prev => ({
      ...prev,
      [column]: value
    }))
  }, [])

  const handleDragEnd = useCallback((event: DragEndEvent) => {
    const { active, over } = event
    if (active.id !== over?.id) {
      setColumnOrder(items => {
        const oldIndex = items.indexOf(active.id.toString())
        const newIndex = items.indexOf(over?.id.toString() ?? '')
        return arrayMove(items, oldIndex, newIndex)
      })
    }
  }, [])

  const filteredAndSortedData = useMemo(() => {
    let result = [...data]

    // Apply filters
    Object.entries(filters).forEach(([column, value]) => {
      if (value) {
        result = result.filter(row => {
          const cellContent = getCellContent(row, column)
          return String(cellContent).toLowerCase().includes(value.toLowerCase())
        })
      }
    })

    // Apply sorting
    if (sortColumn) {
      result.sort((a, b) => {
        const aValue = String(getCellContent(a, sortColumn))
        const bValue = String(getCellContent(b, sortColumn))
        return sortDirection === "asc" 
          ? aValue.localeCompare(bValue)
          : bValue.localeCompare(aValue)
      })
    }

    return result
  }, [data, filters, sortColumn, sortDirection, getCellContent])

  const paginatedData = useMemo(() => {
    const startIndex = (currentPage - 1) * pageSize
    const endIndex = startIndex + pageSize
    return filteredAndSortedData.slice(startIndex, endIndex)
  }, [filteredAndSortedData, currentPage, pageSize])

  const totalPages = Math.ceil(filteredAndSortedData.length / pageSize)

  const handlePrevPage = useCallback(() => {
    setCurrentPage(prev => Math.max(1, prev - 1))
  }, [])

  const handleNextPage = useCallback(() => {
    setCurrentPage(prev => Math.min(totalPages, prev + 1))
  }, [totalPages])

  const handlePageClick = useCallback((page: number) => {
    setCurrentPage(page)
  }, [])

  const orderedColumns = useMemo(() => 
    columnOrder.map(columnKey => 
      columns.find(col => col.key === columnKey)!
    ), [columnOrder, columns]
  )

  const renderCell = (content: React.ReactNode, truncate: boolean) => {
    if (!truncate || typeof content !== 'string') {
      return content
    }

    const maxLength = truncateConfig.maxLength || 100
    if (content.length <= maxLength) {
      return content
    }

    const truncatedContent = `${content.slice(0, maxLength)}...`
    
    return (
      <TooltipProvider>
        <Tooltip>
          <TooltipTrigger asChild>
            <span className="border-b border-dotted border-muted-foreground/50 hover:border-foreground transition-colors">
              {truncatedContent}
            </span>
          </TooltipTrigger>
          <TooltipContent 
            className="max-w-[400px] whitespace-pre-wrap bg-popover/95 backdrop-blur supports-[backdrop-filter]:bg-popover/85"
          >
            {content}
          </TooltipContent>
        </Tooltip>
      </TooltipProvider>
    )
  }

  return (
    <div className="rounded-lg border bg-card">
      <DndContext onDragEnd={handleDragEnd}>
        <Table>
          <TableHeader>
            <TableRow>
              <SortableContext items={columnOrder}>
                {orderedColumns.map(column => (
                  <SortableHeader
                    key={column.key}
                    column={column}
                    onSort={handleSort}
                    sortColumn={sortColumn}
                    sortDirection={sortDirection}
                    onFilter={handleFilter}
                    filterValue={filters[column.key] || ""}
                  />
                ))}
              </SortableContext>
            </TableRow>
          </TableHeader>
          <TableBody>
            {paginatedData.map(row => (
              <motion.tr
                key={getRowKey(row)}
                className={cn(
                  onRowClick ? "cursor-pointer hover:bg-muted/50" : "",
                  rowProps?.(row)?.className
                )}
                onClick={() => onRowClick?.(row)}
                {...(rowProps?.(row) || {})}
              >
                {orderedColumns.map(column => (
                  <TableCell key={`${getRowKey(row)}-${column.key}`}>
                    {renderCell(getCellContent(row, column.key), truncateConfig.enabled)}
                  </TableCell>
                ))}
              </motion.tr>
            ))}
          </TableBody>
        </Table>
      </DndContext>
      
      {totalPages > 1 && (
        <div className="flex justify-center py-4 border-t">
          <Pagination>
            <PaginationContent>
              <PaginationItem>
                <PaginationPrevious 
                  onClick={handlePrevPage}
                  aria-disabled={currentPage === 1}
                />
              </PaginationItem>
              
              {Array.from({ length: totalPages }, (_, i) => i + 1).map((page) => (
                <PaginationItem key={`page-${page}`}>
                  <PaginationLink
                    onClick={() => handlePageClick(page)}
                    isActive={currentPage === page}
                  >
                    {page}
                  </PaginationLink>
                </PaginationItem>
              ))}
              
              <PaginationItem>
                <PaginationNext
                  onClick={handleNextPage}
                  aria-disabled={currentPage === totalPages}
                />
              </PaginationItem>
            </PaginationContent>
          </Pagination>
        </div>
      )}
    </div>
  )
}