"use client"

import { DataItem } from '@/types/dataset'
import { XAxis, YAxis, Tooltip, ResponsiveContainer, Legend, CartesianGrid, Area, ComposedChart } from 'recharts'
import { useState } from 'react'
import { Button } from '@/components/ui/button'

interface TimeSeriesChartProps {
  data: DataItem[]
}

type TimeUnit = 'second' | 'minute' | 'hour'

interface TimePoint {
  timestamp: number
  requestsSent: number
  responsesReceived: number
  timeLabel: string
}

function aggregateByTimeUnit(data: DataItem[], unit: TimeUnit): TimePoint[] {
  const timePoints = new Map<number, TimePoint>()

  // Convert to milliseconds
  const interval = unit === 'second' ? 1000 :
                  unit === 'minute' ? 60 * 1000 :
                  60 * 60 * 1000

  // Find min and max timestamps
  const timestamps = data.flatMap(item => [
    new Date(item.created_at).getTime(),
    new Date(item.finished_at).getTime()
  ])
  const minTime = Math.floor(Math.min(...timestamps) / interval) * interval
  const maxTime = Math.ceil(Math.max(...timestamps) / interval) * interval

  // Initialize all time points
  for (let t = minTime; t <= maxTime; t += interval) {
    timePoints.set(t, {
      timestamp: t,
      requestsSent: 0,
      responsesReceived: 0,
      timeLabel: formatTimeLabel(t, unit)
    })
  }

  // Count requests sent and responses received in each interval
  data.forEach(item => {
    const createdTime = Math.floor(new Date(item.created_at).getTime() / interval) * interval
    const finishedTime = Math.floor(new Date(item.finished_at).getTime() / interval) * interval

    const createdPoint = timePoints.get(createdTime)
    if (createdPoint) {
      createdPoint.requestsSent++
    }

    const finishedPoint = timePoints.get(finishedTime)
    if (finishedPoint) {
      finishedPoint.responsesReceived++
    }
  })

  return Array.from(timePoints.values())
    .sort((a, b) => a.timestamp - b.timestamp)
}

function formatTimeLabel(timestamp: number, unit: TimeUnit): string {
  const date = new Date(timestamp)
  if (unit === 'second') {
    return date.toLocaleTimeString([], {
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit'
    })
  } else if (unit === 'minute') {
    return date.toLocaleTimeString([], {
      hour: '2-digit',
      minute: '2-digit'
    })
  } else {
    return date.toLocaleTimeString([], {
      hour: '2-digit'
    })
  }
}

export function TimeSeriesChart({ data }: TimeSeriesChartProps) {
  const [timeUnit, setTimeUnit] = useState<TimeUnit>('second')
  const timeSeriesData = aggregateByTimeUnit(data, timeUnit)

  return (
    <div className="w-full">
      <div className="flex justify-between items-center mb-4">
        <h2 className="text-xl font-semibold">Requests and Responses Over Time</h2>
        <div className="flex gap-2">
          <Button
            variant={timeUnit === 'second' ? 'default' : 'outline'}
            onClick={() => setTimeUnit('second')}
            size="sm"
          >
            Per Second
          </Button>
          <Button
            variant={timeUnit === 'minute' ? 'default' : 'outline'}
            onClick={() => setTimeUnit('minute')}
            size="sm"
          >
            Per Minute
          </Button>
          <Button
            variant={timeUnit === 'hour' ? 'default' : 'outline'}
            onClick={() => setTimeUnit('hour')}
            size="sm"
          >
            Per Hour
          </Button>
        </div>
      </div>

      <div className="h-80">
        <ResponsiveContainer width="100%" height="100%">
          <ComposedChart
            data={timeSeriesData}
            margin={{ top: 10, right: 30, left: 20, bottom: 15 }}
          >
            <defs>
              <linearGradient id="requestsGradient" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#8884d8" stopOpacity={0.2}/>
                <stop offset="95%" stopColor="#8884d8" stopOpacity={0.05}/>
              </linearGradient>
              <linearGradient id="responsesGradient" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#82ca9d" stopOpacity={0.2}/>
                <stop offset="95%" stopColor="#82ca9d" stopOpacity={0.05}/>
              </linearGradient>
            </defs>
            <CartesianGrid strokeDasharray="3 3" stroke="#888" opacity={0.2} />
            <XAxis
              dataKey="timeLabel"
              fontSize={12}
              angle={-45}
              textAnchor="end"
              height={70}
              stroke="#888"
            />
            <YAxis stroke="#888" />
            <Tooltip
              labelFormatter={(label) => `Time: ${label}`}
              formatter={(value: number, name: string) => [
                value,
                name
              ]}
              contentStyle={{
                backgroundColor: 'var(--tooltip-text-bg)',
                border: '1px solid var(--border)',
                borderRadius: '4px',
                color: 'var(--tooltip-text)'
              }}
              labelStyle={{
                color: 'var(--tooltip-text)'
              }}
            />
            <Legend />
            <Area
              type="monotone"
              dataKey="requestsSent"
              fill="url(#requestsGradient)"
              stroke="#8884d8"
              strokeWidth={2}
              name="Requests Sent"
              dot={false}
            />
            <Area
              type="monotone"
              dataKey="responsesReceived"
              fill="url(#responsesGradient)"
              stroke="#82ca9d"
              strokeWidth={2}
              name="Responses Received"
              dot={false}
            />
          </ComposedChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
