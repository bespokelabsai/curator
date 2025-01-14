import { DataItem } from '@/types/dataset'
import { existsSync, promises as fs } from 'fs'
import { NextRequest, NextResponse } from 'next/server'
import { homedir } from 'os'
import { join } from 'path'

export const dynamic = 'force-dynamic'
export const runtime = 'nodejs'

async function readJsonlFile(filePath: string): Promise<DataItem[]> {
  const content = await fs.readFile(filePath, 'utf-8')
  return content.split('\n')
    .filter(line => line.trim() !== '')
    .map(line => JSON.parse(line))
}

export async function GET(
  request: NextRequest,
  { params }: { params: Promise<{ runHash: string }> }
): Promise<Response> {
  try {
    const { runHash } = await params
    const cacheDir = process.env.CURATOR_CACHE_DIR || join(homedir(), '.cache', 'curator')
    const runDir = join(cacheDir, runHash)

    const searchParams = request.nextUrl.searchParams
    const lastLineNumber = parseInt(searchParams.get('lastLine') || '0')
    const processedFiles = (searchParams.get('processedFiles') || '').split(',').filter(Boolean)
    const isBatchMode = searchParams.get('batchMode') === 'true'

    if (!existsSync(runDir)) {
      return NextResponse.json(
        { error: "Run directory not found" },
        { status: 404 }
      )
    }

    if (isBatchMode) {
      // Batch streaming mode: Read all response files that haven't been processed
      const files = await fs.readdir(runDir)
      const responseFiles = files
        .filter(f => f.startsWith('responses_') && f.endsWith('.jsonl'))
        .filter(f => !processedFiles.includes(f))

      const allData: DataItem[] = []
      for (const file of responseFiles) {
        const filePath = join(runDir, file)
        const fileData = await readJsonlFile(filePath)
        allData.push(...fileData)
      }

      return NextResponse.json({
        data: allData,
        processedFiles: responseFiles,
        isBatchMode: true,
        totalLines: null
      })
    } else {
      // Online streaming mode
      const responsesPath = join(runDir, 'responses_0.jsonl')

      // Check if the file exists before trying to read it
      if (!existsSync(responsesPath)) {
        return NextResponse.json({
          data: [],
          totalLines: 0,
          isBatchMode: false,
          processedFiles: null,
          message: "No responses file found yet"
        })
      }

      const content = await fs.readFile(responsesPath, 'utf-8')
      const lines = content.split('\n').filter(line => line.trim() !== '')

      const newLines = lines.slice(lastLineNumber)
      const jsonData = newLines.map(line => JSON.parse(line))

      return NextResponse.json({
        data: jsonData,
        totalLines: lines.length,
        isBatchMode: false,
        processedFiles: null
      })
    }
  } catch (error) {
    console.error("Error reading responses file:", error)
    return NextResponse.json(
      { error: "Failed to read responses file" },
      { status: 500 }
    )
  }
}
