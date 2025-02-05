"use client"

import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer } from 'recharts'
import { bin } from 'd3-array'
import { DataItem } from '@/types/dataset'

interface DistributionChartProps {
  data: DataItem[]
  column: string
}

const COLUMN_DISPLAY_NAMES: Record<string, string> = {
  total_tokens: "Total Tokens",
  prompt_tokens: "Prompt Tokens",
  completion_tokens: "Completion Tokens",
  generation_time: "Generation Time (s)"
}

function getDistributionData(data: DataItem[], column: string) {
  const values = data.map(item => {
    switch (column) {
      case "total_tokens":
        return item.raw_response.usage?.total_tokens || item.raw_response.response?.body?.usage?.total_tokens || 0;
      case "prompt_tokens":
        return item.raw_response.usage?.prompt_tokens || item.raw_response.response?.body?.usage?.prompt_tokens || 0;
      case "completion_tokens":
        return item.raw_response.usage?.completion_tokens || item.raw_response.response?.body?.usage?.completion_tokens || 0;
      case "generation_time":
        return (new Date(item.finished_at).getTime() - new Date(item.created_at).getTime()) / 1000;
      default:
        return 0;
    }
  });

  const binner = bin()
    .value(d => d)
    .thresholds(10);

  const bins = binner(values);

  return bins.map(b => ({
    range: Number.isInteger(b.x0) ? `${b.x0}` : `${b.x0?.toFixed(2)}`,
    count: b.length,
    start: b.x0,
    end: b.x1
  }));
}

export function DistributionChart({ data, column }: DistributionChartProps) {
  const distributionData = getDistributionData(data, column);

  if (distributionData.length === 0) return null;

  const displayName = COLUMN_DISPLAY_NAMES[column] || column;

  return (
    <div className="w-full h-80">
      <h2 className="text-xl font-semibold mb-4">Distribution of {displayName}</h2>
      <ResponsiveContainer width="100%" height="100%">
        <BarChart
          data={distributionData}
          margin={{ top: 10, right: 30, left: 20, bottom: 15 }}
        >
          <XAxis
            dataKey="range"
            fontSize={12}
            angle={-45}
            textAnchor="end"
            height={70}
          />
          <YAxis />
          <Tooltip
            formatter={(value: number) => [`Count: ${value}`, displayName]}
            contentStyle={{
              backgroundColor: 'var(--tooltip-text-bg)',
              border: '1px solid var(--border)',
              borderRadius: '4px',
              color: 'var(--tooltip-text)',
              boxShadow: '0 2px 4px rgba(0,0,0,0.1)'
            }}
            labelStyle={{
              color: 'var(--tooltip-text)',
              marginBottom: '4px'
            }}
            cursor={{ fill: 'var(--tooltip-bg)', opacity: 0.2 }}
          />
          <Bar
            dataKey="count"
            fill="#8884d8"
            name={displayName}
          />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
