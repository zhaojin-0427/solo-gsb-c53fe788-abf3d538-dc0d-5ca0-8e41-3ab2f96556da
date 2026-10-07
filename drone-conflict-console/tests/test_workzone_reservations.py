"""Temporary work-zone reservation semantics and storage arbitration.

Covers (pure logic + SQLite, no HTTP layer):

* happy path: a start safe under the start-window query is registered and
  reloads with the same id/parameters and status "safe";
* submission re-adjudication against the LATEST state INSIDE one write
  transaction:
    - illegal parameters / s outside [t, T-d] -> 400, no row;
    - unknown batch -> 404, no row;
    - t earlier than a reported actual take-off -> 400
      ``t_before_reported_takeoff``;
    - overdue sorties -> 409 ``overdue_unconfirmable``;
    - closed action interval meeting continuous-time cylinder occupancy
      (including an endpoint-only touch) -> 409 ``start_occupied``;
    - same-batch space-time intersection (time OR cylinder merely touching a
      boundary already conflicts) -> 409 ``reservation_conflict``;
      disjoint cylinders at the same time, and the same cylinder at a
      disjoint time, are both accepted;
  every refusal rolls back and leaves no reservation behind;
* later reports never revoke a reservation: the list re-judges on the newest
  state and marks a now-occupied one ``affected`` and one whose t now precedes
  a reported actual take-off (or with an overdue sortie) ``unconfirmable``;
* deterministic two-process races:
    - a take-off report committed in the query-submission gap is seen by the
      reservation transaction and an outdated start is refused, no row;
    - two reservations committed for the same space-time overlap leave only
      the winner, the loser is refused with the winner's id;
* deleting a batch removes its reservations.

Run:  python tests/test_workzone_reservations.py
"""

from __future__ import annotations

import multiprocessing as mp
import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.batch import Report  # noqa: E402
from app.model import Waypoint, make_sortie  # noqa: E402
from app.reservation import (  # noqa: E402
    ReservationRejected,
    cylinders_intersect,
    evaluate_reservation_status,
    reservations_space_time_conflict,
)
from app.storage import Store  # noqa: E402

FAILS = 0


def check(cond, msg):
    global FAILS
    if not cond:
        FAILS += 1
        print("FAIL:", msg)
    else:
        print("ok:", msg)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def eastbound(sid="A", x0=0, x1=100, z=20, dur=10):
    return make_sortie(sid, 1, 30, [
        Waypoint(x0, 0, z, 0), Waypoint(x1, 0, z, dur)])


def make_store_with_batch(path: str | None = None, offsets=("A", 0)):
    """Fresh store + scenario + batch with an eastbound sortie at plan 0."""
    tmp = None
    if path is None:
        tmp = tempfile.TemporaryDirectory()
        path = os.path.join(tmp.name, "t.db")
    store = Store(path)
    sid = store.create_scenario({"sorties": [{"id": "A"}]}, 30, 10)
    plan = dict([offsets])
    store.save_result(sid, {
        "feasible": True, "status": "optimal",
        "weighted_delay": sum(plan.values()), "offsets": plan,
        "locked": {}, "elapsed_seconds": 0.01, "solved_conflicts": [],
        "cores": [], "truncated": False,
    }, 30, 10)
    bid = store.create_batch(sid, "b")
    sorties = [eastbound("A")]
    return store, bid, sorties, tmp


# Cylinder over the track at (50,0), r=10, band [10,30]; an A-on-plan-0
# eastbound sortie occupies global [4,6].  t=0,T=12,d=2 -> safe starts are
# [0,1] and [7,10].
ZONE = dict(cx=50, cy=0, radius=10, z_lo=10, z_hi=30, t=0, T=12, d=2, s=7)


# ---------------------------------------------------------------------------
# Pure geometry
# ---------------------------------------------------------------------------


def test_cylinder_intersection():
    # Side-by-side disks touching at one x: distance == r1 + r2 -> conflict.
    check(cylinders_intersect(0, 0, 10, 10, 20, 20, 0, 10, 10, 20),
          "horizontally touching cylinders intersect")
    check(not cylinders_intersect(0, 0, 10, 10, 20, 20 + 1e-3, 0, 10, 10, 20),
          "disjoint disks do not intersect")
    # Altitude bands meeting at one plane: zhi1 == zlo2 -> conflict.
    check(cylinders_intersect(0, 0, 10, 10, 20, 0, 0, 10, 20, 30),
          "altitude bands meeting at a plane intersect")
    check(not cylinders_intersect(0, 0, 10, 10, 20, 0, 0, 10, 20 + 1e-3, 30),
          "separated altitude bands do not intersect")
    # Same centre but one band above the other separated -> no conflict.
    check(not cylinders_intersect(0, 0, 10, 0, 10, 0, 0, 10, 11, 20),
          "vertical gap keeps cylinders apart")


def test_space_time_conflict():
    r1 = {"center_x": 0, "center_y": 0, "radius": 10, "z_lo": 0, "z_hi": 10,
          "start": 0, "duration": 4}
    r2 = dict(r1, start=4)  # [0,4] touches [4,8]
    check(reservations_space_time_conflict(r1, r2),
          "time intervals touching at an endpoint conflict")
    r3 = dict(r1, start=5)  # gap of one second
    check(not reservations_space_time_conflict(r1, r3),
          "disjoint time intervals do not conflict")
    r4 = dict(r1, center_x=21)  # same time, genuinely disjoint cylinders
    check(not reservations_space_time_conflict(r1, r4),
          "disjoint cylinders at the same time do not conflict")
    r5 = dict(r1, center_x=21)  # touching cylinders AND disjoint time
    check(not reservations_space_time_conflict(dict(r1, start=5), r5),
          "no time overlap -> no space-time conflict even with overlap")
    r6 = dict(r1, center_x=21, start=4)
    check(not reservations_space_time_conflict(r1, r6),
          "touching time but disjoint cylinders -> no conflict")
    r7 = {"center_x": 0, "center_y": 0, "radius": 10, "z_lo": 11, "z_hi": 20,
          "start": 0, "duration": 4}
    check(not reservations_space_time_conflict(r1, r7),
          "overlapping time and disk but separated altitude bands -> no conflict")
    # Touching in BOTH dimensions (horizontal centres exactly r1+r2 apart,
    # time intervals sharing one endpoint) already conflicts.
    rt = dict(r1, center_x=20, start=4)
    check(reservations_space_time_conflict(r1, rt),
          "boundary touching in cylinder and time conflicts")


# ---------------------------------------------------------------------------
# Store: creation happy path and refusals
# ---------------------------------------------------------------------------


def _expect_reject(fn, status, reason=None):
    try:
        fn()
    except ReservationRejected as exc:
        ok = exc.status_code == status and (reason is None or exc.reason == reason)
        check(ok, f"expected {status} {reason}, got {exc.status_code} {exc.reason}")
        return exc
    check(False, f"expected rejection {status} {reason}")
    return None


def test_create_and_reload():
    store, bid, sorties, tmp = make_store_with_batch()
    try:
        r = store.create_reservation(sorties, bid, **ZONE)
        check(r["id"] == 1 and r["start"] == 7 and r["duration"] == 2,
              f"inserted reservation: {r}")
        rows = store.list_reservation_rows(bid)
        check(len(rows) == 1 and rows[0]["T"] == 12 and rows[0]["t"] == 0,
              "listed reservation keeps t/T/d/s and the cylinder")
        check(store.get_reservation_row(bid, 1)["center_x"] == 50,
              "get reservation by id")
        check(store.get_reservation_row(bid, 999) is None,
              "unknown reservation id -> None")
        check(store.get_reservation_row(bid + 999, 1) is None,
              "reservation of another (missing) batch -> None")
    finally:
        tmp.cleanup()


def test_unknown_batch():
    tmp = tempfile.TemporaryDirectory()
    try:
        store = Store(os.path.join(tmp.name, "t.db"))
        sorties = [eastbound("A")]
        _expect_reject(
            lambda: store.create_reservation(sorties, 4242, **ZONE),
            404, "batch_not_found")
    finally:
        tmp.cleanup()


def test_illegal_parameters_leave_no_row():
    store, bid, sorties, tmp = make_store_with_batch()
    try:
        cases = [
            ("radius", dict(radius=0)),
            ("band", dict(z_lo=30, z_hi=20)),
            ("T<t", dict(t=5, T=4)),
            ("d=0", dict(d=0)),
            ("s<t", dict(s=-1)),
            ("s>T-d", dict(s=11)),  # T-d = 10
        ]
        for label, over in cases:
            kw = {**ZONE, **over}
            # keep s inside the (possibly altered) candidate window where the
            # parameter-level reason must be the one that fires
            if label == "T<t":
                kw["s"] = 4
            _expect_reject(
                lambda kw=kw: store.create_reservation(sorties, bid, **kw),
                400)
        check(store.list_reservation_rows(bid) == [],
              "illegal requests left no reservations")

        # Boundary: s == T-d is legal (action ends exactly at T).
        r = store.create_reservation(sorties, bid, **{**ZONE, 's': 10})
        check(r["start"] == 10, "s = T-d accepted")
        # s == t legal too (that one is also safe here).
        r0 = store.create_reservation(
            sorties, bid, **{**ZONE, "s": 0, "cx": 500, "cy": 500})
        check(r0["start"] == 0, "s = t accepted")
    finally:
        tmp.cleanup()


def test_start_occupied_and_touching():
    store, bid, sorties, tmp = make_store_with_batch()
    try:
        # Occupancy [4,6], d=2: s=2 -> [2,4] touches occupancy at 4.
        _expect_reject(
            lambda: store.create_reservation(sorties, bid, **{**ZONE, 's': 2}),
            409, "start_occupied")
        # Clearly intersecting.
        _expect_reject(
            lambda: store.create_reservation(sorties, bid, **{**ZONE, 's': 5}),
            409, "start_occupied")
        # Safe start after the track.
        r = store.create_reservation(sorties, bid, **ZONE)
        check(r["start"] == 7, "safe start s=7 after occupancy accepted")
        check(len(store.list_reservation_rows(bid)) == 1,
              "only the safe reservation was stored")
    finally:
        tmp.cleanup()


def test_overdue_and_early_t():
    # An unreported sortie with plan 0 at t=3 is overdue -> 409, no row.
    store, bid, sorties, tmp = make_store_with_batch()
    try:
        _expect_reject(
            lambda: store.create_reservation(
                sorties, bid, **{**ZONE, "s": 3, "t": 3, "T": 12}),
            409, "overdue_unconfirmable")
        # A reported take-off later than t -> 400 t_before_reported_takeoff.
        store.submit_report(bid, "A", 9, 1)
        _expect_reject(
            lambda: store.create_reservation(
                sorties, bid, **{**ZONE, "s": 3, "t": 3, "T": 12}),
            400, "t_before_reported_takeoff")
        check(store.list_reservation_rows(bid) == [],
              "overdue/early rejections left no reservations")
        # Report at 4, query at t=4: track occupies [8,10]; s=10 -> [10,12]
        # touches at 10 (forbidden); s=11 with T=13 -> [11,13] safe.
        store.submit_report(bid, "A", 4, 2)
        _expect_reject(
            lambda: store.create_reservation(
                sorties, bid, **{**ZONE, "t": 4, "s": 10}),
            409, "start_occupied")
        r = store.create_reservation(
            sorties, bid, **{**ZONE, "t": 4, "T": 13, "s": 11})
        check(r["start"] == 11, "post-report safe start accepted")
    finally:
        tmp.cleanup()


def test_reservation_space_time_conflicts():
    store, bid, sorties, tmp = make_store_with_batch()
    try:
        far = {**ZONE, "cx": 500, "cy": 500}  # cylinder never occupied
        r1 = store.create_reservation(sorties, bid, **{**far, 's': 0})
        check(r1["id"] == 1, "first reservation")
        # Same cylinder, touching time [0,2] vs [2,4] -> conflict.
        _expect_reject(
            lambda: store.create_reservation(sorties, bid, **{**far, "s": 2}),
            409, "reservation_conflict")
        # Same cylinder, disjoint time [3,5] -> accepted.
        r3 = store.create_reservation(sorties, bid, **{**far, "s": 3})
        check(r3["id"] == 2, "disjoint-time reservation accepted")
        # Same time as #1, horizontally disjoint cylinder -> accepted.
        r4 = store.create_reservation(
            sorties, bid, **{**far, "s": 0, "cx": 600})
        check(r4["id"] == 3, "disjoint-cylinder reservation accepted")
        # Cylinder horizontally touching r4 (centres 100 apart, r=10 each)
        # at the same time -> conflict.
        exc = _expect_reject(
            lambda: store.create_reservation(
                sorties, bid, **{**far, "s": 0, "cx": 620}),
            409, "reservation_conflict")
        check(exc is not None and exc.fields["conflicting_reservation_ids"] == [3],
              f"conflict names the winner id: {exc.fields if exc else None}")
        # Altitude bands meeting at a plane with overlapping time -> conflict.
        _expect_reject(
            lambda: store.create_reservation(
                sorties, bid, **{**far, "s": 0, "z_lo": 30, "z_hi": 40}),
            409, "reservation_conflict")
        check(len(store.list_reservation_rows(bid)) == 3,
              "exactly the three accepted reservations stored")
    finally:
        tmp.cleanup()


# ---------------------------------------------------------------------------
# Live status after later reports (reservations are never revoked)
# ---------------------------------------------------------------------------


def _dto_status(store, bid, sorties, rid):
    row = store.get_batch(bid)
    reports = {r["sortie_id"]: Report(r["actual_offset"], r["revision"])
               for r in row["reports"]}
    rows = store.list_reservation_rows(bid)
    res = next(r for r in rows if r["id"] == rid)
    others = [r for r in rows if r["id"] != rid]
    return evaluate_reservation_status(
        sorties, row["plan"]["offsets"], reports, res, others)


def test_live_status_after_reports():
    store, bid, sorties, tmp = make_store_with_batch()
    try:
        # Query at t=4 with A already reported on plan (actual 0), so it is
        # confirmable and occupies [4,6].
        store.submit_report(bid, "A", 0, 1)
        qt4 = {**ZONE, "t": 4, "T": 13}
        far = {**qt4, "cx": 500, "cy": 500}
        r = store.create_reservation(sorties, bid, **{**far, "s": 4})
        rid = r["id"]
        st = _dto_status(store, bid, sorties, rid)
        check(st["status"] == "safe", "newly created reservation is safe")

        # Near cylinder: at t=4 the on-plan track (actual 0) occupies [4,6];
        # reservation [10,12] is safe.  A reported +4 delay (revision 2,
        # actual 4, >= t so still confirmable) shifts occupancy to [8,10],
        # which touches the action interval at 10.
        near = store.create_reservation(sorties, bid, **{**qt4, "s": 10})
        nid = near["id"]
        store.submit_report(bid, "A", 4, 2)
        st = _dto_status(store, bid, sorties, nid)
        check(st["status"] == "affected"
              and [h["sortie_id"] for h in st["occupancy_conflicts"]] == ["A"],
              f"delay into the action interval marks affected: {st}")
        # The far reservation stays safe (cylinder never near the track).
        check(_dto_status(store, bid, sorties, rid)["status"] == "safe",
              "far-away reservation unaffected by the report")
        # A correction back to plan restores safe.
        store.submit_report(bid, "A", 0, 3)
        check(_dto_status(store, bid, sorties, nid)["status"] == "safe",
              "correction restores safe")

        # t-earlier-than-reported-take-off: the reservation was submitted at
        # t=4; a later report of actual take-off 9 -> unconfirmable.
        store.submit_report(bid, "A", 9, 4)
        st = _dto_status(store, bid, sorties, nid)
        check(st["status"] == "unconfirmable"
              and st["early_violations"] == [{"sortie_id": "A",
                                              "actual_takeoff": 9}],
              f"t=4 before reported take-off 9 -> unconfirmable: {st}")
    finally:
        tmp.cleanup()


def test_live_status_overdue():
    # Two sorties: reservation at t=0 while both on plan; then report A and
    # let B (plan 0) become overdue at the stored t=0? Overdueness is judged
    # at the reservation's stored t, so it cannot arise post-hoc for t=0.
    # Instead: reservation submitted at t=4 with a planned sortie B at 8
    # (predictable then); B later... remaining unreported cannot become
    # overdue relative to t=4.  Overdue status arises when the stored t
    # itself makes an unreported planned sortie overdue - which creation
    # forbids - so the live overdue branch is reachable only defensively.
    tmp = tempfile.TemporaryDirectory()
    try:
        store = Store(os.path.join(tmp.name, "t.db"))
        sid = store.create_scenario({"sorties": [{"id": "A"}, {"id": "B"}]},
                                    30, 10)
        plan = {"A": 0, "B": 8}
        store.save_result(sid, {
            "feasible": True, "status": "optimal", "weighted_delay": 8,
            "offsets": plan, "locked": {}, "elapsed_seconds": 0.01,
            "solved_conflicts": [], "cores": [], "truncated": False}, 30, 10)
        bid = store.create_batch(sid, "b")
        sorties = [eastbound("A"), eastbound("B")]
        # A reported on plan (actual 0 <= t=4); B still planned at 8 >= 4.
        store.submit_report(bid, "A", 0, 1)
        far = {**ZONE, "cx": 500, "cy": 500, "t": 4}
        r = store.create_reservation(sorties, bid, **{**far, 's': 4})
        # The live overdue branch cannot arise from later reports alone (a
        # stored t that had an overdue sortie was never accepted at creation);
        # exercise the pure function on a synthesized row at t=9 where B
        # (plan 8, still unreported) is overdue.
        synth = dict(r, t=9)
        row = store.get_batch(bid)
        reports = {x["sortie_id"]: Report(x["actual_offset"], x["revision"])
                   for x in row["reports"]}
        st = evaluate_reservation_status(
            sorties, plan, reports, synth, [r])
        check(st["status"] == "unconfirmable"
              and st["overdue_sorties"] == ["B"],
              f"overdue at the stored t -> unconfirmable: {st}")
    finally:
        tmp.cleanup()


def test_batch_delete_cascades():
    store, bid, sorties, tmp = make_store_with_batch()
    try:
        store.create_reservation(
            sorties, bid, **{**ZONE, "s": 0, "cx": 500, "cy": 500})
        check(len(store.list_reservation_rows(bid)) == 1, "one reservation")
        check(store.delete_batch(bid), "batch deleted")
        check(store.list_reservation_rows(bid) == [],
              "reservations removed with the batch")
    finally:
        tmp.cleanup()


# ---------------------------------------------------------------------------
# Deterministic two-process races
# ---------------------------------------------------------------------------


def _reserve_worker(path, bid, ready, gate, queue, kwargs):
    store = Store(path)
    sorties = [eastbound("A")]
    ready.set()
    gate.wait()
    try:
        r = store.create_reservation(sorties, bid, **kwargs)
        queue.put(("ok", r["id"]))
    except ReservationRejected as exc:
        queue.put(("rejected", exc.status_code, exc.reason, exc.fields))
    except Exception as exc:  # a lock failure must not masquerade as success
        queue.put(("error", repr(exc)))


def test_report_commits_in_gap():
    """A take-off report committed while the reservation waits for the write
    lock must be re-adjudicated inside the reservation transaction: the safe
    start under the query-time state becomes occupied and is refused."""
    ctx = mp.get_context("fork")
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "t.db")
        store, bid, sorties, _ = make_store_with_batch(path)
        # Pre-query state: A reported on plan at actual 0 occupies [4,6];
        # s=7 with t=4 is safe.  Hold an uncommitted correction (revision 2,
        # actual 4 -> occupancy [8,10]) so the child's BEGIN IMMEDIATE queues;
        # the reservation [7,9] then intersects it.
        ready, gate = ctx.Event(), ctx.Event()
        queue = ctx.Queue()
        store.submit_report(bid, "A", 0, 1)
        sql = ("UPDATE batch_reports SET actual_offset=?,revision=?,reported_at=? "
               "WHERE batch_id=? AND sortie_id=?",
               (4, 2, time.time(), bid, "A"))
        child = ctx.Process(target=_reserve_worker,
                            args=(path, bid, ready, gate, queue,
                                  {**ZONE, "t": 4, "T": 13, "s": 7}))
        child.start()
        ready.wait(30)  # child Store initialized
        import sqlite3
        raw = sqlite3.connect(path, timeout=10, isolation_level=None)
        raw.execute("PRAGMA busy_timeout=10000")
        # Parent opens the pending report write FIRST, then releases the gate:
        # the child's BEGIN IMMEDIATE is guaranteed to queue on this lock, so
        # its authoritative in-transaction read happens after our commit.
        raw.execute("BEGIN IMMEDIATE")
        raw.execute(*sql)
        gate.set()
        time.sleep(0.6)  # child queues on the write lock in this window
        raw.commit()
        raw.close()
        outcome = queue.get(timeout=30)
        child.join(10)
        check(child.exitcode == 0, f"child finished cleanly -> {outcome}")
        check(outcome[0] == "rejected" and outcome[1] == 409
              and outcome[2] == "start_occupied",
              f"report committed first: outdated start refused 409 -> {outcome}")
        check(Store(path).list_reservation_rows(bid) == [],
              "gap report refusal left no reservation")
        # After the report, s=7 is genuinely unsafe via a fresh request too.
        try:
            store.create_reservation(
                sorties, bid, **{**ZONE, "t": 4, "T": 13, "s": 7})
            check(False, "post-commit adjudication must also refuse s=7")
        except ReservationRejected as exc:
            check(exc.status_code == 409 and exc.reason == "start_occupied",
                  "fresh request sees the committed report")


def test_two_reservations_race():
    """Two processes submit touching time intervals for the same far
    cylinder.  The write lock serializes them: one wins, the other is
    refused with the winner's id; exactly one row remains."""
    ctx = mp.get_context("fork")
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "t.db")
        store, bid, _, _ = make_store_with_batch(path)
        far = {**ZONE, "cx": 500, "cy": 500}

        # Launch both gates near-simultaneously so their BEGIN IMMEDIATE calls
        # race (SQLite serializes internally).
        ready1, gate1, ready2, gate2 = (ctx.Event(), ctx.Event(),
                                        ctx.Event(), ctx.Event())
        q1, q2 = ctx.Queue(), ctx.Queue()
        p1 = ctx.Process(target=_reserve_worker,
                         args=(path, bid, ready1, gate1, q1, {**far, "s": 0}))
        p2 = ctx.Process(target=_reserve_worker,
                         args=(path, bid, ready2, gate2, q2, {**far, "s": 2}))
        p1.start(); p2.start()
        ready1.wait(30); ready2.wait(30)
        gate1.set(); gate2.set()
        o1, o2 = q1.get(timeout=30), q2.get(timeout=30)
        p1.join(10); p2.join(10)
        outs = [o1, o2]
        wins = [o for o in outs if o[0] == "ok"]
        loses = [o for o in outs if o[0] == "rejected"]
        check(len(wins) == 1 and len(loses) == 1,
              f"exactly one winner/one loser -> {outs}")
        if wins and loses:
            wid = wins[0][1]
            check(loses[0][1] == 409 and loses[0][2] == "reservation_conflict"
                  and loses[0][3]["conflicting_reservation_ids"] == [wid],
                  f"loser names winner #{wid} -> {loses[0]}")
        rows = Store(path).list_reservation_rows(bid)
        check(len(rows) == 1, f"exactly one reservation survives: {len(rows)}")


def main():
    test_cylinder_intersection()
    test_space_time_conflict()
    test_create_and_reload()
    test_unknown_batch()
    test_illegal_parameters_leave_no_row()
    test_start_occupied_and_touching()
    test_overdue_and_early_t()
    test_reservation_space_time_conflicts()
    test_live_status_after_reports()
    test_live_status_overdue()
    test_batch_delete_cascades()
    test_report_commits_in_gap()
    test_two_reservations_race()
    print("reservation checks, failures:", FAILS)
    return FAILS


if __name__ == "__main__":
    sys.exit(1 if main() else 0)
