import { DatasetViewer } from "@/components/dataset-viewer/DatasetViewer"

export default async function DatasetPage({
  params,
  searchParams
}: {
  params: Promise<{ runHash: string }>,
  searchParams: Promise<{ [key: string]: string | string[] | undefined }>
}) {
  const { runHash } = await params
  const { batchMode } = await searchParams
  const isBatchMode = batchMode === '1'

  return <DatasetViewer runHash={runHash} batchMode={isBatchMode} />
}
