import { NextResponse } from 'next/server'
import { Database } from 'sqlite3'
import { homedir } from 'os'
import { join } from 'path'
import { existsSync } from 'fs'

export async function GET(request: Request): Promise<Response>  {
  return new Promise((resolve) => {
    const { searchParams } = new URL(request.url)
    const lastCreatedTime = searchParams.get('lastCreatedTime')

    const cacheDir = process.env.CURATOR_CACHE_DIR || join(homedir(), '.cache', 'curator')
    const dbPath = join(cacheDir, 'metadata.db')

    if (!existsSync(dbPath)) {
      console.error(`Database file not found at: ${dbPath}`)
      resolve(NextResponse.json(
        {
          error: 'NO_CACHE_DB',
          message: 'No cache database found. Please run some prompts first.',
          path: dbPath
        },
        { status: 404 }
      ))
      return
    }

    try {
      const db = new Database(dbPath, (err) => {
        if (err) {
          console.error('Database connection error:', err)
          resolve(NextResponse.json({ error: 'Database connection failed' }, { status: 500 }))
          return
        }
      })

      const query = lastCreatedTime
        ? 'SELECT * FROM runs WHERE created_time > ? ORDER BY created_time DESC'
        : 'SELECT * FROM runs ORDER BY created_time DESC'

      const params = lastCreatedTime ? [lastCreatedTime] : []

      db.all(
        query,
        params,
        (err, rows) => {
          if (err) {
            console.error('Database query error:', err)
            resolve(NextResponse.json({ error: 'Database query failed' }, { status: 500 }))
            return
          }

          const safeRows = Array.isArray(rows) ? rows : []
          resolve(NextResponse.json(safeRows))

          db.close((closeErr) => {
            if (closeErr) {
              console.error('Error closing database:', closeErr)
            }
          })
        }
      )
    } catch (error) {
      console.error('Unexpected error:', error)
      resolve(NextResponse.json({ error: 'Unexpected error occurred' }, { status: 500 }))
    }
  })
}
