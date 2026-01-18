from typing import List, Optional, Literal
from pydantic import BaseModel, Field


Severity = Literal["SEV1", "SEV2", "SEV3"]
IncidentStatus = Literal["open", "resolved"]
TicketStatus = Literal["open", "resolved"]
Priority = Literal["P1", "P2", "P3"]


class Incident(BaseModel):
    id: str
    title: str
    service: str
    severity: Severity
    status: IncidentStatus
    created_at: str
    resolved_at: Optional[str] = None
    root_cause: Optional[str] = None
    tags: List[str] = Field(default_factory=list)


class Ticket(BaseModel):
    id: str
    type: str
    title: str
    service: str
    priority: Priority
    status: TicketStatus
    created_at: str
    sla_due_at: Optional[str] = None
