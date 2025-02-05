import { ReactNode } from 'react'

export type SortDirection = "asc" | "desc"

export interface Column {
  key: string
  label: string
}

export interface TruncateConfig {
  enabled: boolean
  maxLength: number
}

export interface SortableHeaderProps {
  column: Column
  onSort: (columnKey: string) => void
  sortColumn: string | null
  sortDirection: SortDirection
  onFilter: (columnKey: string, value: string) => void
  filterValue: string
}

export interface SortableTableProps {
  columns: Column[]
  data: any[]
  getRowKey: (item: any) => string | number
  getCellContent: (item: any, columnKey: string) => ReactNode
  onRowClick?: (item: any) => void
  truncateConfig?: TruncateConfig
  initialSortColumn?: string
  initialSortDirection?: SortDirection
  pageSize?: number
  rowProps?: (row: any) => { className?: string }
}
