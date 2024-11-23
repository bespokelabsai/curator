import { homedir } from 'os'
import { join } from 'path'

const CURATOR_DEFAULT_CACHE_DIR = '~/.cache/curator'

export function getCacheDir(): string {
  // Check environment variable first
  const envCacheDir = process.env.CURATOR_CACHE_DIR
  if (envCacheDir) {
    return envCacheDir.startsWith('~') 
      ? join(homedir(), envCacheDir.slice(1))
      : envCacheDir
  }

  // Fallback to default cache directory
  return join(homedir(), '.cache', 'curator')
} 