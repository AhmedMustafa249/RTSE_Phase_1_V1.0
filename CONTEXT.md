# RTSE Phase 1 — Hack2Drive 2026 — Glossary

## Domain terms

**Lane** — One of 5 fixed horizontal road positions (index 0=leftmost .. 4=rightmost). The car always
snaps to and stays centered in whichever lane it's currently in; there is no continuous lane drift to
correct for.

**Tap (steering tap)** — The only way to move between lanes. A short pulse of full steering input
(±1.0) followed by a return to 0.0. Holding steering does not work; the sim expects a discrete pulse.
Empirically ~100ms or less triggers a lane change.

**Hazard** — A red or yellow token detected ahead of the car, within the reaction zone, close enough
to the car's current lane to threaten a collision. Both colors are treated as **equally bad** (yellow's
randomized debuffs are considered as costly as red's speed penalty) — there is no scoring/weighting
between them, just binary avoid/no-avoid.

**Hard-avoid rule** — The MVP decision policy: a lane is either legal (no hazard in it) or illegal (hazard
present). The car picks any legal lane, with no score-summing across tokens. This is a deliberate
rejection of [[drive_with_detector-scoring-bug]] found in a colleague's reference implementation, where
summing weighted token scores per lane caused the car to sometimes steer toward a red token because a
nearby green outweighed it in the sum.

**Reaction zone** — The vertical band of the front-camera frame (roughly y=225 to y=430 in a 640x480
frame) where tokens are close enough to the car to require a decision. Tokens above this band are too
far away to act on yet.

**Static blob** — A detected red/yellow-colored region that is NOT a real token: a fixed road marking,
UI element, or other artifact that sits in the same screen position frame after frame. Distinguished
from a real token by consecutive-frame persistence in the same bucketed position; suppressed once seen
too many times in a row so it can't be mistaken for a hazard. Real tokens move (scroll toward the car)
and don't trigger this filter. This idea was borrowed from the colleague's `drive_with_detector.py`
(`StaticBlobTracker`) after a false-positive static blob caused rapid oscillating lane changes.

## Reference material (not authoritative, ideas only)

`rtse_framework.py`, `detector.py`, `drive_with_detector.py`, `diag_run.py` — a colleague's existing
Linux-targeted implementation (see git log "chore: add back all py files from Ahmed Galeb's Linux repo").
Decision made: **write our own detector and driving logic from scratch**, based on `sample_drive.py`'s
RTOS structure, not by importing/extending these files — both to avoid inheriting the scoring bug and
because each team member needs to demonstrate individual understanding of their own code for assessment.
