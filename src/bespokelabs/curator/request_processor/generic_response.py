from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class GenericResponse(BaseModel):
    response: Optional[Dict[str, Any]] | str = None
    request: Optional[Dict[str, Any]] = None
    errors: Optional[List[str]] = None
    row: Optional[Dict[str, Any]] = None
    row_idx: int
    raw_response: Optional[Dict[str, Any]] = None
