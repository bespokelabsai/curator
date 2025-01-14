export type Message = {
  role: string
  content: string
}

export type DataItem = {
  response_message: string | Record<string, any>
  response_errors: string[] | null
  raw_response: null | any
  raw_request: null | any
  generic_request: {
    model: string
    messages: Message[]
    response_format: Record<string, any> | null
    original_row: Record<string, any>
    original_row_idx: number
  }
  created_at: string
  finished_at: string
}

export interface Run {
  run_hash: string;
  dataset_hash: string;
  prompt_func: string;
  model_name: string;
  response_format: string;
  created_time: string;
  last_edited_time: string;
}
