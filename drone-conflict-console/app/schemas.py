"""Pydantic schemas and import-time validation."""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator, model_validator


class WaypointIn(BaseModel):
    x: float
    y: float
    z: float
    t: int = Field(..., description="seconds from take-off; strictly increasing, >= 0")

    @field_validator("t")
    @classmethod
    def t_nonneg(cls, v: int) -> int:
        if v < 0:
            raise ValueError("waypoint time must be >= 0")
        return v


class SortieIn(BaseModel):
    id: str = Field(..., min_length=1)
    weight: int = Field(..., gt=0, description="positive integer priority weight")
    window_hi: int = Field(..., ge=0, description="latest take-off offset in seconds")
    waypoints: list[WaypointIn] = Field(..., min_length=2)

    @field_validator("id")
    @classmethod
    def id_nonblank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("sortie id must not be blank")
        return v

    @model_validator(mode="after")
    def _check(self):
        ts = [w.t for w in self.waypoints]
        if any(b <= a for a, b in zip(ts, ts[1:])):
            raise ValueError("waypoint times must be strictly increasing integers")
        if self.window_hi < 0:
            raise ValueError("window_hi must be >= 0")
        return self


class ImportPayload(BaseModel):
    sorties: list[SortieIn] = Field(..., min_length=1)
    safety_radius: float = Field(..., gt=0)
    safety_height: float = Field(..., gt=0)

    @model_validator(mode="after")
    def _unique_ids(self):
        ids = [s.id for s in self.sorties]
        if len(set(ids)) != len(ids):
            dup = sorted({i for i in ids if ids.count(i) > 1})
            raise ValueError(f"duplicate sortie ids: {dup}")
        return self


class LockEntry(BaseModel):
    sortie_id: str
    offset: int = Field(..., ge=0)


class SolveRequest(BaseModel):
    safety_radius: float | None = Field(None, gt=0)
    safety_height: float | None = Field(None, gt=0)
    locks: list[LockEntry] = Field(default_factory=list)
    time_limit: float = Field(20.0, gt=0, le=120)


class SaveRequest(BaseModel):
    name: str = Field(..., min_length=1)


class CreateBatchRequest(BaseModel):
    name: str = Field("", description="optional display name for the batch")
    use_stored_plan: bool = Field(
        False,
        description="explicit acknowledgment that the batch freezes the plan "
                    "currently stored in SQLite (the retained old table with "
                    "its original thresholds) rather than a freshly verified "
                    "solve; REQUIRED when the scenario's latest solve returned "
                    "unknown and a feasible old table is retained",
    )
    confirmed_safety_radius: float | None = Field(
        None, gt=0,
        description="the old table's original safety radius the client is "
                    "confirming; required (and must match the stored value) "
                    "together with use_stored_plan after an unknown solve",
    )
    confirmed_safety_height: float | None = Field(
        None, gt=0,
        description="the old table's original safety height the client is "
                    "confirming; required (and must match the stored value) "
                    "together with use_stored_plan after an unknown solve",
    )


class ReportRequest(BaseModel):
    sortie_id: str = Field(..., min_length=1)
    actual_offset: int = Field(
        ..., ge=0, description="actual take-off offset in seconds (non-negative)")
    revision: int = Field(
        ..., ge=1, description="report revision, 1 first then strictly increasing")


class WorkZoneReservationRequest(BaseModel):
    """One concrete work-zone start-second reservation.

    The domain checks shared with the start-window query (positive radius,
    z_lo < z_hi, non-negative t, T >= t, positive d) and the candidate-window
    membership ``s ∈ [t, T-d]`` are re-adjudicated by the server against the
    AUTHORITATIVE batch state inside the submission write transaction -
    trusting client-side checks alone cannot be safe across processes.
    """

    cx: float
    cy: float
    radius: float = Field(..., gt=0)
    z_lo: float
    z_hi: float
    t: int = Field(..., ge=0)
    T: int = Field(..., description="deadline second (integer, >= t)")
    d: int = Field(..., gt=0, description="positive integer action duration")
    s: int = Field(..., ge=0,
                   description="chosen integer start second in [t, T-d]")

    @model_validator(mode="after")
    def _check(self):
        if self.z_lo >= self.z_hi:
            raise ValueError("z_lo must be strictly less than z_hi")
        if self.T < self.t:
            raise ValueError("deadline T must not be earlier than t")
        if not (self.t <= self.s <= self.T - self.d):
            raise ValueError(
                f"start s must be an integer in [t, T-d] "
                f"([{self.t}, {self.T - self.d}])")
        return self
