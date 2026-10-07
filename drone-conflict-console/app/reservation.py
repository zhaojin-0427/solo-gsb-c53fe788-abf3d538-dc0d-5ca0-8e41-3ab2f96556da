"""Temporary work-zone reservations on an execution batch.

A *reservation* registers one concrete integer start second ``s`` for a work
action of duration ``d`` inside a vertical cylindrical protection zone,
submitted together with the non-negative current second ``t`` and the deadline
second ``T`` used at submission time.

The adjudication at submission time, all performed against the AUTHORITATIVE
batch/report rows restored from SQLite, is:

1. the same parameter legality checks as the start-window query
   (:func:`app.workzone.validate_workzone_query`);
2. ``s`` must be an integer in ``[t, T - d]``;
3. ``t`` must not precede any reported actual take-off (same rule as the
   warning / work-zone query);
4. no participating sortie may be overdue: an overdue sortie's future track
   is unknowable, so the start cannot be confirmed;
5. the closed action interval ``[s, s + d]`` must not intersect ANY
   continuous-time cylinder occupancy of the effective trajectories;
6. it must not space-time intersect any reservation already registered for
   the SAME batch.

Two reservations intersect in space-time exactly when their closed time
intervals overlap AND their closed cylinders overlap - touching only at a
boundary (``s1 + d1 == s2``, horizontal centres exactly ``r1 + r2`` apart,
altitude bands meeting at one plane) already counts as a conflict.

The status of a stored reservation is always RE-COMPUTED from the latest
reports when it is listed or reloaded (later reports never revoke a
reservation):

* ``safe`` - still confirmable and still free of occupancy/inter-reservation
  conflicts;
* ``affected`` - confirmable, but the latest reports move a trajectory into
  the action interval or another registered reservation now space-time
  intersects it;
* ``unconfirmable`` - an overdue sortie exists, or the reservation's own
  submission ``t`` is now earlier than a reported actual take-off.

All checks at creation time run inside one ``BEGIN IMMEDIATE`` write
transaction together with the insert (see :meth:`Store.create_reservation`),
so a report revision committed in the query-submission gap or another
reservation committing first can never make an outdated start pass: the write
lock serializes report writes, reservation writes and the authoritative read.
A refusal rolls the transaction back and leaves no row behind.
"""

from __future__ import annotations

from .batch import (
    EarlyTimeReject,
    Report,
    classify_sorties,
)
from .geometry import EPS
from .workzone import (
    InvalidWorkZoneQuery,
    cylinder_occupancy_segments,
    validate_workzone_query,
)

#: Status at creation time / while still safe under the newest reports.
SAFE = "safe"
#: Confirmable, but the newest reports or a newer reservation make it unsafe.
AFFECTED = "affected"
#: An overdue sortie exists or its t now precedes a reported actual take-off.
UNCONFIRMABLE = "unconfirmable"

#: Two boundaries closer than this are treated as touching (closed sets):
#: touching already counts as a space-time conflict, resolved on the safe
#: side.
_TOUCH_EPS = 1e-7


class ReservationRejected(Exception):
    """A reservation request refused against the AUTHORITATIVE batch state.

    Raised inside the same ``BEGIN IMMEDIATE`` transaction that would insert
    the reservation, so the decision is always taken on the committed rows
    visible at write time (never on a snapshot taken before the transaction)
    and the rollback guarantees a refusal leaves no reservation behind.
    ``status_code`` is the HTTP status the route returns, ``reason`` is the
    machine-readable cause and ``fields`` are merged into the response
    ``detail`` object.
    """

    def __init__(self, status_code: int, reason: str, fields: dict | None = None):
        self.status_code = status_code
        self.reason = reason
        self.fields = fields or {}
        super().__init__(f"work-zone reservation rejected: {reason}")


# ---------------------------------------------------------------------------
# Pure geometry: closed-cylinder / closed-time intersection of reservations
# ---------------------------------------------------------------------------


def cylinders_intersect(cx1: float, cy1: float, r1: float, zlo1: float, zhi1: float,
                        cx2: float, cy2: float, r2: float, zlo2: float, zhi2: float
                        ) -> bool:
    """Closed vertical cylinders overlap (boundary contact counts).

    Horizontal overlap: centre distance ``<= r1 + r2`` (distance compared
    squared to avoid a sqrt).  Vertical overlap: the closed altitude bands
    share at least one plane, i.e. ``zlo1 <= zhi2`` and ``zlo2 <= zhi1``.
    """
    dx, dy = cx1 - cx2, cy1 - cy2
    rr = float(r1) + float(r2)
    horizontal = dx * dx + dy * dy <= rr * rr + _TOUCH_EPS
    vertical = (float(zlo1) <= float(zhi2) + _TOUCH_EPS
                and float(zlo2) <= float(zhi1) + _TOUCH_EPS)
    return horizontal and vertical


def reservations_space_time_conflict(a: dict, b: dict) -> bool:
    """Whether two stored/incoming reservation rows space-time intersect.

    The closed time intervals ``[s, s + d]`` overlap (touching counts) AND
    the closed cylinders overlap.  ``a`` / ``b`` are plain mappings carrying
    ``center_x/center_y/radius/z_lo/z_hi/start/duration`` (or the
    ``cx/cy/radius`` aliases of an incoming request).
    """
    def cyl(r):
        return (
            r.get("center_x", r.get("cx")), r.get("center_y", r.get("cy")),
            r.get("radius"), r.get("z_lo"), r.get("z_hi"),
        )

    if int(a["start"]) + int(a["duration"]) < int(b["start"]) - _TOUCH_EPS:
        return False
    if int(b["start"]) + int(b["duration"]) < int(a["start"]) - _TOUCH_EPS:
        return False
    return cylinders_intersect(*cyl(a), *cyl(b))


# ---------------------------------------------------------------------------
# Trajectory occupancy at a candidate start
# ---------------------------------------------------------------------------


def occupancy_hits_start(sorties, statuses: dict[str, str],
                         effective: dict[str, int],
                         cx: float, cy: float, radius: float,
                         z_lo: float, z_hi: float,
                         s: int, d: int,
                         T: int | None = None) -> list[dict]:
    """Participating sorties whose cylinder occupancy meets ``[s, s+d]``.

    The occupancy is computed in continuous time per segment; only instants
    in the closed action interval ``[s, s + d]`` (optionally clipped at the
    deadline ``T``) count.  Overdue sorties are skipped by the caller's
    classification.  Returns one descriptor per hitting sortie with the
    occupancy section(s) intersecting the action interval.
    """
    lo, hi = float(s), float(s + d)
    if T is not None:
        hi = min(hi, float(T))
    hits: list[dict] = []
    for sortie in sorties:
        sid = sortie.id
        if statuses[sid] == "overdue":
            continue
        ivs = cylinder_occupancy_segments(
            sortie, effective[sid], cx, cy, radius, z_lo, z_hi)
        meet = [(a, b) for a, b in ivs if b >= lo - EPS and a <= hi + EPS]
        if meet:
            hits.append({
                "sortie_id": sid,
                "status": statuses[sid],
                "takeoff_offset": effective[sid],
                "occupied_intervals": [
                    {"time_start": round(max(a, lo), 6),
                     "time_end": round(min(b, hi), 6)}
                    for a, b in meet],
            })
    return hits


def _row_values(r) -> tuple:
    return (r["center_x"], r["center_y"], r["radius"],
            r["z_lo"], r["z_hi"])


def conflicting_reservations(candidate: dict,
                             existing: list[dict]) -> list[dict]:
    """Stored reservations (same batch) space-time intersecting ``candidate``."""
    return [r for r in existing if reservations_space_time_conflict(candidate, r)]


# ---------------------------------------------------------------------------
# Submission adjudication (used inside the storage write transaction)
# ---------------------------------------------------------------------------


def adjudicate_reservation(sorties, planned_offsets: dict[str, int],
                           reports: dict[str, Report],
                           cx: float, cy: float, radius: float,
                           z_lo: float, z_hi: float,
                           t: int, T: int, d: int, s: int,
                           existing: list[dict]) -> dict:
    """Validate a reservation request against one committed batch state.

    Raises :class:`ReservationRejected` (HTTP-shaped) for every refusal;
    otherwise returns a dict with the resolved classification context
    (statuses/effective/occupancy hits/intersecting reservations), which the
    storage layer uses for the insert and the response.
    """
    try:
        validate_workzone_query(cx, cy, radius, z_lo, z_hi, t, T, d)
    except InvalidWorkZoneQuery as exc:
        raise ReservationRejected(400, exc.reason, exc.fields)

    if not isinstance(s, int) or isinstance(s, bool):
        raise ReservationRejected(400, "s_not_integer", {"s": s})
    cand_lo, cand_hi = t, T - d
    if s < cand_lo or s > cand_hi:
        raise ReservationRejected(
            400, "start_outside_window",
            {"s": s, "t": t, "T": T, "duration": d,
             "candidate_window": {"start_lo": cand_lo, "start_hi": cand_hi,
                                  "has_candidates": cand_lo <= cand_hi}})

    try:
        statuses, effective = classify_sorties(
            sorties, planned_offsets, reports, t)
    except EarlyTimeReject as exc:
        raise ReservationRejected(
            400, "t_before_reported_takeoff",
            {"t": t, "violations": exc.violations})

    overdue_ids = [sid for sid, st in statuses.items() if st == "overdue"]
    if overdue_ids:
        raise ReservationRejected(
            409, "overdue_unconfirmable",
            {"t": t, "overdue_sorties": overdue_ids})

    hits = occupancy_hits_start(
        sorties, statuses, effective,
        cx, cy, radius, z_lo, z_hi, s, d, T)
    if hits:
        raise ReservationRejected(
            409, "start_occupied",
            {"s": s, "duration": d,
             "action_interval": [s, s + d],
             "occupancy_conflicts": hits})

    candidate = {
        "center_x": cx, "center_y": cy, "radius": radius,
        "z_lo": z_lo, "z_hi": z_hi, "start": s, "duration": d,
    }
    intersecting = conflicting_reservations(candidate, existing)
    if intersecting:
        raise ReservationRejected(
            409, "reservation_conflict",
            {"s": s, "duration": d,
             "action_interval": [s, s + d],
             "conflicting_reservation_ids": [int(r["id"]) for r in intersecting]})

    return {"statuses": statuses, "effective": effective}


# ---------------------------------------------------------------------------
# Live status of a stored reservation under the newest committed reports
# ---------------------------------------------------------------------------


def evaluate_reservation_status(sorties, planned_offsets: dict[str, int],
                                reports: dict[str, Report],
                                reservation: dict,
                                others: list[dict]) -> dict:
    """Recompute a stored reservation's status from the latest committed state.

    Later reports never revoke a reservation: its parameters stay as stored,
    only the live ``status`` / reasons change.  ``others`` are the remaining
    stored reservations of the same batch (each itself safe at creation).

    Returns ``{"status", "overdue_sorties", "early_violations",
    "occupancy_conflicts", "conflicting_reservation_ids"}``.
    """
    t, T, d, s = (reservation["t"], reservation["T"],
                  reservation["duration"], reservation["start"])

    early = [
        {"sortie_id": sid, "actual_takeoff": rep.actual_offset}
        for sid, rep in sorted(reports.items())
        if t < rep.actual_offset
    ]
    overdue_ids: list[str] = []
    statuses: dict[str, str] = {}
    effective: dict[str, int] = {}
    if not early:
        # Same classification as everywhere else; an EarlyTimeReject here is
        # already captured by ``early`` above.
        statuses, effective = classify_sorties(
            sorties, planned_offsets, reports, t)
        overdue_ids = [sid for sid, st in statuses.items() if st == "overdue"]

    if early or overdue_ids:
        return {"status": UNCONFIRMABLE,
                "overdue_sorties": overdue_ids,
                "early_violations": early,
                "occupancy_conflicts": [],
                "conflicting_reservation_ids": []}

    cx, cy = reservation["center_x"], reservation["center_y"]
    radius = reservation["radius"]
    z_lo, z_hi = reservation["z_lo"], reservation["z_hi"]
    hits = occupancy_hits_start(
        sorties, statuses, effective,
        cx, cy, radius, z_lo, z_hi, s, d, T)
    conflicts = [int(r["id"]) for r in others
                 if reservations_space_time_conflict(reservation, r)]
    status = AFFECTED if (hits or conflicts) else SAFE
    return {"status": status,
            "overdue_sorties": [],
            "early_violations": [],
            "occupancy_conflicts": hits,
            "conflicting_reservation_ids": sorted(conflicts)}
