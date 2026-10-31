"""The payment that is supposed to put drones alongside a suspect vessel.

The escort credit was a squared ramp and the fleet stopped 1.47 km short of a
3 km radius, drifting two thirds of the time it was on station. The reason is a
property of the shape rather than of its size: `(1 - g/R)^2` has gradient
`2(1-g/R)/R`, which vanishes at the rim. The rim is where a vehicle arrives, so
at the one moment it needed to be told to keep coming the term fell silent --
and holding position costs no fuel.

These tests hold the shape that replaced it. Three of them are about the
gradient, because the gradient is what was wrong; the last one is the
behavioural claim, checked through the real credit rather than the formula.

What is NOT tested here is the weight. 16 was priced by scoring candidate
payments over two recorded trajectories, and that measurement lives in
`config/world.yaml` beside the number it chose.
"""

from __future__ import annotations

import math
from dataclasses import replace

import pytest
import torch

from baltic_isr import config
from baltic_isr.config import Config, Reward
from baltic_isr.fleet import auv as auv_module
from baltic_isr.learn.env import Environment, EnvState
from baltic_isr.picture.tracks import Tracks


@pytest.fixture(scope="module")
def resolved() -> Config:
    return config.load("baseline")


def _flag(state: EnvState, vessel_index: int) -> Tracks:
    """Return tracks where every vehicle holds an actionable verdict on one vessel.

    `_suspected` is a conjunction -- flagged, held, believed slow, not stale --
    so a test that sets only `deviant` is testing against an empty set and will
    pass whatever the code does.
    """
    flagged = torch.zeros_like(state.tracks.deviant)
    flagged[:, :, vessel_index] = True
    return replace(
        state.tracks,
        deviant=flagged,
        last_seen_s=torch.full_like(state.tracks.last_seen_s, state.t),
        velocity=torch.zeros_like(state.tracks.velocity),
        cov=torch.zeros_like(state.tracks.cov),
    )


def pays(settings: Reward, gap_m: float) -> float:
    """Return what the escort pays a lone vehicle at this range, from the config."""
    floor = math.exp(-settings.shadow_radius_m / settings.shadow_hug_m)
    core = (math.exp(-gap_m / settings.shadow_hug_m) - floor) / (1.0 - floor)
    return float(settings.shadow) * max(core, 0.0)


def test_the_escort_pays_nothing_at_the_rim(resolved: Config) -> None:
    """The one virtue of the ramp it replaced, kept deliberately.

    A bare exponential would pay for hanging about at the edge of the radius,
    which is the wide-circle failure this project has already filmed once. The
    core is shifted by its own rim value so the edge is worth exactly nothing.
    """
    settings = resolved.reward
    assert pays(settings, settings.shadow_radius_m) == pytest.approx(0.0)


def test_closing_always_pays(resolved: Config) -> None:
    """A gradient at every range inside the radius, the rim included.

    This is the defect being fixed, stated as a test. The squared ramp's
    marginal value of closing a hundred metres falls to 0.013 at 2.9 km of a
    3 km radius -- indistinguishable from nothing against a reward whose terms
    run to 0.88 -- so a vehicle that had just arrived was paid the same for
    stopping as for continuing.
    """
    settings = resolved.reward
    rim = settings.shadow_radius_m
    step = 100.0
    ranges = torch.arange(0.0, rim, step).tolist()
    margins = [pays(settings, g) - pays(settings, g + step) for g in ranges]

    assert min(margins) > 0.0, "some range pays nothing for closing"
    # And the rim is not a special case: the hundred metres just inside it is
    # worth more than the whole reward's smallest meaningful term.
    assert pays(settings, rim - step) - pays(settings, rim) > 0.01

    old = [
        settings.shadow * (1.0 - g / rim) ** 2
        - settings.shadow * (1.0 - min(g + step, rim) / rim) ** 2
        for g in ranges
    ]
    assert margins[-1] > old[-1], "the new shape is no steeper at the rim than the old one"


def test_the_last_kilometre_is_worth_more_than_the_first(resolved: Config) -> None:
    """Progressive, in the sense that was asked for: more the closer it is glued.

    Measured on the configured numbers: 0.71 at 2 km, 3.66 at 1 km, 10.35 at
    300 m. Coming in from 2 km to 1 km earns 2.95 a decision, and the last
    700 m of that earns another 6.69.
    """
    settings = resolved.reward
    far = pays(settings, 1000.0) - pays(settings, 2000.0)
    near = pays(settings, 300.0) - pays(settings, 1000.0)
    assert near > far * 2.0


def test_the_rim_pays_less_than_a_vehicle_earns_elsewhere(resolved: Config) -> None:
    """The invariant that makes loitering at the rim pointless by arithmetic.

    Cable patrol pays about 0.88 a decision, measured off-station. If sitting
    at the edge of a suspect's circle paid more than that, a vehicle could do
    better than its day job by doing nothing in particular near a flagged
    vessel -- which is how the fleet came to loiter in wide circles before. The
    configured profile pays 0.71 there.

    The comparison is against `cable`, the term's own weight, because that is
    the quantity in the config rather than a figure from one run: a rim payment
    below a fifth of the cable weight leaves the margin the measurement found.
    """
    settings = resolved.reward
    rim_ish = pays(settings, 0.67 * settings.shadow_radius_m)
    assert rim_ish < settings.cable / 5.0


def test_only_the_team_is_paid(resolved: Config) -> None:
    """The cap is a cliff, not a taper, and it is still a cliff.

    The sixth vehicle on a contact earns nothing at all rather than a little
    less. A taper was tried and seventeen vehicles converged on one contact.
    """
    settings = resolved.reward
    env = Environment(resolved, 1, "cpu")
    state, _ = env.reset(seed=5)

    vessel = state.traffic.xy[0, 0]
    crowd = int(settings.shadow_team) + 3
    moved = state.fleet.xy.clone()
    # Line them up at increasing range, all well inside the radius, so the
    # ranking is unambiguous and every one of them would be paid but for the
    # cap.
    for i in range(crowd):
        moved[0, i] = vessel + torch.tensor([100.0 * (i + 1), 0.0])
    fleet = type(state.fleet)(
        **{
            **vars(state.fleet),
            "xy": moved,
            "mode": torch.full_like(state.fleet.mode, auv_module.MODE_DEPLOYED),
        }
    )

    suspected = torch.zeros_like(state.tracks.deviant)
    suspected[0, :, 0] = True
    # Through the real assignment, because the cap IS the assignment now: a
    # credit test that hands `_shadow` a cap of its own would be testing
    # arithmetic nobody runs.
    tracks = _flag(state, vessel_index=0)
    claim = env._claims(
        torch.full((1, env.n_vehicles), -1, dtype=torch.int16),
        state.t,
        fleet,
        state.traffic,
        tracks,
    )
    credit = env._shadow(env._suspected(state.t, tracks), fleet, state.traffic, claim)

    assert int((claim[0] == 0).sum()) == int(settings.shadow_team)
    assert float(credit[0, : int(settings.shadow_team)].min()) > 0.0
    nobody = credit[0, int(settings.shadow_team) : crowd]
    assert float(nobody.max()) == pytest.approx(0.0)


def test_the_cap_is_a_held_claim_and_not_a_fresh_ranking(resolved: Config) -> None:
    """Five may be paid for a vessel, and the sixth gets nothing -- every look.

    This is the defect that survived two training runs. The cap was a RANK,
    recomputed three times a decision; with ten drones milling at similar
    ranges each one spent part of its time in the first five, so the surplus
    collected 1.335 a decision of a term meant to pay it nothing, and staying
    beat patrolling 1.087 to 0.968. A cap that is re-drawn is not exclusion,
    it is a lottery, and rationing a big pot still draws a mob.

    Capping the SHAPING could never have fixed it either: potential-based
    shaping leaves the optimal policy unchanged whatever the potential is.
    Only the credit decides who comes.

    Ten vehicles are placed on one suspect. The claim must name five, the same
    five on the next call, and the credit must reach exactly those.
    """
    settings = resolved.reward
    team = int(settings.shadow_team)
    env = Environment(resolved, 1, "cpu")
    state, _ = env.reset(seed=11)

    vessel = state.traffic.xy[0, 0]
    crowd = team + 5
    moved = state.fleet.xy.clone()
    for i in range(crowd):
        moved[0, i] = vessel + torch.tensor([400.0 * (i + 1), 0.0])
    moved[0, crowd:] = vessel + torch.tensor([60_000.0, 0.0])
    fleet = type(state.fleet)(
        **{
            **vars(state.fleet),
            "xy": moved,
            "mode": torch.full_like(state.fleet.mode, auv_module.MODE_DEPLOYED),
        }
    )
    tracks = _flag(state, vessel_index=0)
    empty = torch.full((1, env.n_vehicles), -1, dtype=torch.int16)

    claim = env._claims(empty, state.t, fleet, state.traffic, tracks)
    assert int((claim[0] == 0).sum()) == team, "the cap is not being honoured"

    # Held, not re-drawn: the same call again returns the same five.
    again = env._claims(claim, state.t, fleet, state.traffic, tracks)
    assert torch.equal(claim, again), "the claim churns between calls"

    # And the attraction pulls the claimants only, so shaping and credit agree.
    pulled = env._suspect_potential(replace(state, fleet=fleet, tracks=tracks, escort_claim=claim))
    assert int((pulled[0] > 0.0).sum()) == team


def test_a_vessel_never_collects_more_than_its_team(resolved: Config) -> None:
    """The cap holds over a run, not just in a contrived frame.

    The assignment carries claims forward and fills free slots, and the way
    that goes wrong is a slot being granted twice in one pass -- a vehicle that
    is the best candidate for two vessels at once. So this drives the real
    environment and checks the invariant every decision.
    """
    team = int(resolved.reward.shadow_team)
    env = Environment(resolved, 2, "cpu")
    state, _ = env.reset(seed=5)
    actions = torch.zeros((2, env.n_vehicles), dtype=torch.long)

    worst = 0
    for _ in range(12):
        state, _ = env.step(state, actions)
        claim = state.escort_claim.long()
        for world in range(claim.shape[0]):
            held = claim[world][claim[world] >= 0]
            if held.numel():
                worst = max(worst, int(held.bincount().max()))
    assert worst <= team, f"a vessel collected {worst} escorts at a cap of {team}"


def test_being_alongside_beats_being_in_the_zone(resolved: Config) -> None:
    """The behavioural claim, through the real credit rather than the formula.

    Two fleets, same suspect, same everything else: one at 2.5 km, inside the
    radius and earning; one at 500 m. If the second did not earn several times
    the first, the term would not express "converge on it" however steep the
    algebra looked.
    """
    env = Environment(resolved, 1, "cpu")
    state, _ = env.reset(seed=7)
    vessel = state.traffic.xy[0, 0]

    suspected = torch.zeros_like(state.tracks.deviant)
    suspected[0, :, 0] = True

    def credit_at(offset_m: float) -> float:
        moved = state.fleet.xy.clone()
        moved[0, 0] = vessel + torch.tensor([offset_m, 0.0])
        fleet = type(state.fleet)(
            **{
                **vars(state.fleet),
                "xy": moved,
                "mode": torch.full_like(state.fleet.mode, auv_module.MODE_DEPLOYED),
            }
        )
        tracks = _flag(state, vessel_index=0)
        claim = env._claims(
            torch.full((1, env.n_vehicles), -1, dtype=torch.int16),
            state.t,
            fleet,
            state.traffic,
            tracks,
        )
        return float(
            env._shadow(env._suspected(state.t, tracks), fleet, state.traffic, claim)[0, 0]
        )

    in_the_zone = credit_at(2500.0)
    alongside = credit_at(500.0)
    assert in_the_zone > 0.0, "the zone has to pay something or there is no approach"
    assert alongside > in_the_zone * 5.0


def test_the_wait_is_charged_and_the_escort_is_not(resolved: Config) -> None:
    """The term that empties the queue must not touch the five doing the work.

    This is the whole safety property of the charge. The escort cap was exact
    and the crowd was still eleven, because exclusion stops the sixth vehicle
    being paid and not from waiting for a vacancy. The charge falls on having
    no claim -- so a claimed escort pays nothing however close it sits, and the
    only vehicles it can reach are ones escorting nothing.
    """
    settings = resolved.reward
    assert settings.shadow_crowding > 0.0, "the charge is switched off"
    env = Environment(resolved, 1, "cpu")
    state, _ = env.reset(seed=3)

    vessel = state.traffic.xy[0, 0]
    crowd = int(settings.shadow_team) + 4
    moved = state.fleet.xy.clone()
    for i in range(crowd):
        moved[0, i] = vessel + torch.tensor([300.0 * (i + 1), 0.0])
    # And one far away, doing its day job, which must be untouched.
    moved[0, crowd] = vessel + torch.tensor([60_000.0, 0.0])
    fleet = type(state.fleet)(
        **{
            **vars(state.fleet),
            "xy": moved,
            "mode": torch.full_like(state.fleet.mode, auv_module.MODE_DEPLOYED),
        }
    )
    tracks = _flag(state, vessel_index=0)
    claim = env._claims(
        torch.full((1, env.n_vehicles), -1, dtype=torch.int16),
        state.t,
        fleet,
        state.traffic,
        tracks,
    )
    suspected = env._suspected(state.t, tracks)
    charge = env._crowding(suspected, fleet, state.traffic, claim)
    credit = env._shadow(suspected, fleet, state.traffic, claim)

    team = int(settings.shadow_team)
    assert float(charge[0, :team].max()) == pytest.approx(0.0), "an escort is being charged"
    assert float(charge[0, team:crowd].min()) == pytest.approx(settings.shadow_crowding)
    assert float(charge[0, crowd]) == pytest.approx(0.0), "patrol is being charged"
    # The two halves disagree about nobody: paid XOR charged, on station.
    assert bool(((credit[0, :crowd] > 0.0) ^ (charge[0, :crowd] > 0.0)).all())


def test_waiting_is_worse_than_going_back_to_work(resolved: Config) -> None:
    """The charge has to beat the opportunity cost it is correcting, not just exist.

    Priced over two recorded seeds and 477 waits, the discounted edge of
    queueing beside a full vessel over returning to patrol was +1.66, and
    crossed zero at a charge of 0.35 a decision. The configured 1.0 is a
    threefold margin on a figure measured from one policy.

    Bounded above by what being alongside actually pays rather than by the
    weight, because that is the quantity a charge could overwhelm: a term that
    grew to a comparable size would deter the approach instead of the
    loitering, which is the failure this project has already filmed once.
    """
    settings = resolved.reward
    assert settings.shadow_crowding > 0.35 * 2.0, "no margin over the measured threshold"
    assert settings.shadow_crowding < settings.cable, "the charge outweighs the work it diverts to"
    assert settings.shadow_crowding < pays(settings, 300.0) / 5.0, "big enough to deter the escort"


def test_a_slot_survives_the_transit_it_was_granted_for(resolved: Config) -> None:
    """A vehicle called from 10 km must still hold its slot when it is 10 km out.

    The bug this pins down survived a training run because the test beside it
    put every vehicle inside the 3 km radius, where claims were already stable.
    Slots are GRANTED out to `shadow_reach_m` so a vessel can call in a vehicle
    that has not arrived yet, and were RELEASED at `shadow_radius_m` -- so the
    slot was taken back on the next decision, refilled by whoever was nearest,
    and the fresh ranking the claim exists to replace was still running
    everywhere outside the payment radius.

    Measured consequences: the attraction flickered for everyone inbound, so
    more vehicles were drawn in than there were slots, and the five places went
    to the five that arrived first rather than the five that were tasked.
    """
    settings = resolved.reward
    team = int(settings.shadow_team)
    env = Environment(resolved, 1, "cpu")
    state, _ = env.reset(seed=13)

    vessel = state.traffic.xy[0, 0]
    # The tasked five, strung out along the approach between the payment radius
    # and the reach -- every one of them beyond 3 km, which is the whole point.
    far = [4_000.0, 6_000.0, 8_000.0, 9_500.0, 11_000.0][:team]
    moved = state.fleet.xy.clone()
    for i, range_m in enumerate(far):
        moved[0, i] = vessel + torch.tensor([range_m, 0.0])
    # And a crowd that is NEARER but was not tasked. Under the defect these
    # evicted the inbound five by simply being closer.
    for j in range(team):
        moved[0, team + j] = vessel + torch.tensor([500.0 * (j + 1), 0.0])
    moved[0, 2 * team :] = vessel + torch.tensor([60_000.0, 0.0])
    fleet = type(state.fleet)(
        **{
            **vars(state.fleet),
            "xy": moved,
            "mode": torch.full_like(state.fleet.mode, auv_module.MODE_DEPLOYED),
        }
    )
    tracks = _flag(state, vessel_index=0)

    # Grant the five inbound vehicles their slots by hand, as a decision
    # further out would have done, then ask what survives.
    granted = torch.full((1, env.n_vehicles), -1, dtype=torch.int16)
    granted[0, :team] = 0
    kept = env._claims(granted, state.t, fleet, state.traffic, tracks)

    assert torch.equal(kept[0, :team], granted[0, :team]), "the transit lost its slot"
    assert int((kept[0] == 0).sum()) == team, "the cap broke while the five were inbound"
    assert int((kept[0, team : 2 * team] == 0).sum()) == 0, "a nearer vehicle evicted an incumbent"

    # And the attraction follows the claim, so the five being pulled are the
    # five that were tasked rather than the five that happen to be closest.
    pulled = env._suspect_potential(replace(state, fleet=fleet, tracks=tracks, escort_claim=kept))
    assert int((pulled[0] > 0.0).sum()) == team
    assert bool((pulled[0, :team] > 0.0).all()), "the inbound five are not being pulled"


def test_a_slot_is_released_once_the_vehicle_gives_up_on_it(resolved: Config) -> None:
    """Held on the reach, not held for ever -- the other half of the same rule.

    Releasing on the reach rather than the payment radius is only safe if the
    reach still releases. A slot kept from the far side of the gulf would be a
    vessel escorted by nobody while its five places were spoken for, which is
    worse than the churn it replaced.
    """
    settings = resolved.reward
    env = Environment(resolved, 1, "cpu")
    state, _ = env.reset(seed=13)

    vessel = state.traffic.xy[0, 0]
    moved = state.fleet.xy.clone()
    # Just inside the reach, and just outside it.
    moved[0, 0] = vessel + torch.tensor([settings.shadow_reach_m * 0.95, 0.0])
    moved[0, 1] = vessel + torch.tensor([settings.shadow_reach_m * 1.05, 0.0])
    moved[0, 2:] = vessel + torch.tensor([60_000.0, 0.0])
    fleet = type(state.fleet)(
        **{
            **vars(state.fleet),
            "xy": moved,
            "mode": torch.full_like(state.fleet.mode, auv_module.MODE_DEPLOYED),
        }
    )
    tracks = _flag(state, vessel_index=0)

    granted = torch.full((1, env.n_vehicles), -1, dtype=torch.int16)
    granted[0, 0] = 0
    granted[0, 1] = 0
    kept = env._claims(granted, state.t, fleet, state.traffic, tracks)

    assert int(kept[0, 0]) == 0, "a vehicle inside the reach lost its slot"
    assert int(kept[0, 1]) == -1, "a vehicle beyond the reach kept its slot"


def test_a_pile_is_charged_and_a_radio_link_is_not(resolved: Config) -> None:
    """The charge begins past the paid team and stops short of the modem.

    These are the two halves of one rule: being able to hear each other is
    worth paying for, sharing the same square of water is not. The modem
    reaches 6 km and the charge stops at 3, so the band between them -- where
    the fleet relays without piling up -- costs nothing.
    """
    environment = Environment(resolved, worlds=1, device="cpu")
    state, _ = environment.reset(seed=5)
    fleet = state.fleet
    settings = resolved.reward
    assert settings.packing_radius_m < resolved.comms.range_m

    xy = fleet.xy.clone()
    mode = fleet.mode.clone()
    team = settings.packing_allowance
    mode[0, : team + 3] = auv_module.MODE_DEPLOYED
    xy[0, : team + 3] = xy[0, 0]
    # One more hull just outside the charge but well inside the modem.
    xy[0, team + 3] = xy[0, 0] + torch.tensor([settings.packing_radius_m + 100.0, 0.0])
    mode[0, team + 3] = auv_module.MODE_DEPLOYED

    idle = torch.full_like(fleet.mode, -1, dtype=torch.int16)
    charged = environment._packing(replace(fleet, xy=xy, mode=mode), idle)[0]
    surplus = (team + 3) - 1 - team
    assert charged[0] == pytest.approx(settings.packing_weight * surplus)
    assert charged[team + 3] == 0.0


def test_the_charge_is_symmetric_rather_than_a_ranking(resolved: Config) -> None:
    """Everyone in the knot pays the same, and a lone vehicle pays nothing.

    An earlier attempt to thin a crowd kept the nearest few and charged the
    rest. That ranking was re-drawn every decision, so it taught the policy
    nothing. This one points the same way for every hull in the pile and lets
    what each is earning decide who leaves.
    """
    environment = Environment(resolved, worlds=1, device="cpu")
    state, _ = environment.reset(seed=5)
    fleet = state.fleet
    settings = resolved.reward

    xy = fleet.xy.clone()
    mode = fleet.mode.clone()
    size = settings.packing_allowance + 4
    mode[0, :size] = auv_module.MODE_DEPLOYED
    xy[0, :size] = xy[0, 0]
    # A vehicle on its own, far from the pile and from everyone else.
    far = xy[0, 0] + torch.tensor([200_000.0, 200_000.0])
    xy[0, size] = far
    mode[0, size] = auv_module.MODE_DEPLOYED

    idle = torch.full_like(fleet.mode, -1, dtype=torch.int16)
    charged = environment._packing(replace(fleet, xy=xy, mode=mode), idle)[0]
    assert len(set(charged[:size].tolist())) == 1
    assert charged[0] > 0.0
    assert charged[size] == 0.0


def test_an_escort_is_never_charged_for_the_crowd_it_is_part_of(resolved: Config) -> None:
    """Holding a claim exempts a vehicle from the packing charge.

    The first version of this term charged everyone in the knot, and the cost
    was measurable: escort effort spent on a vessel that was actually
    deviating fell from 83.3 % to 54.1 %. A crowded real rover became
    expensive while a lonely false positive stayed free, so the fleet drifted
    towards isolated suspects. The vehicles asked to leave a pile are the ones
    not working in it.
    """
    environment = Environment(resolved, worlds=1, device="cpu")
    state, _ = environment.reset(seed=5)
    fleet = state.fleet
    settings = resolved.reward

    xy = fleet.xy.clone()
    mode = fleet.mode.clone()
    size = settings.packing_allowance + 4
    mode[0, :size] = auv_module.MODE_DEPLOYED
    xy[0, :size] = xy[0, 0]
    packed = replace(fleet, xy=xy, mode=mode)

    idle = torch.full_like(fleet.mode, -1, dtype=torch.int16)
    assert environment._packing(packed, idle)[0, 0] > 0.0

    # The same pile, but the first few hold a slot on a vessel.
    claim = idle.clone()
    claim[0, : settings.shadow_team] = 0
    charged = environment._packing(packed, claim)[0]
    assert (charged[: settings.shadow_team] == 0.0).all()
    assert charged[settings.shadow_team] > 0.0
