# Human-in-the-Loop Safety

Every optimization run is a recommendation and is created with:

```text
status = awaiting_approval
requires_human_approval = true
```

Only an awaiting-approval run can be approved or rejected. A second decision
returns a conflict. The reviewer, notes, and review timestamp are retained in
the run record and a versioned WebSocket event is emitted.

The current implementation does not dispatch vehicles. Future dispatch and
material reallocation endpoints must require an approved run and create an
immutable audit entry.

