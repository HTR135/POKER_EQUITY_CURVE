#!/usr/bin/env python3

import re
import itertools
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import matplotlib.pyplot as plt
import pandas as pd

HERO = "HL1350"
INPUT_FILE = "hand_history4.txt"

RE_HAND_SPLIT = re.compile(r"(?=PokerStars Hand #\d+:)")
RE_HEADER = re.compile(
    r"PokerStars Hand #(?P<hand_id>\d+):\s+Hold'em No Limit\s+\(\$(?P<sb>[0-9.]+)/\$(?P<bb>[0-9.]+) USD\)"
)
RE_TABLE_BTN = re.compile(r"Table '([^']+)' 6-max Seat #(\d+) is the button")
RE_SEAT_STACK = re.compile(r"Seat (\d+): ([^(]+) \(\$([0-9.]+) in chips\)")
RE_HERO_SEAT = re.compile(
    rf"Seat (\d+): {re.escape(HERO)} \(\$([0-9.]+) in chips\)")
RE_DEALT = re.compile(rf"Dealt to {re.escape(HERO)} \[([^\]]+)\]")
RE_RAKE = re.compile(r"Rake \$([0-9.]+)")

RE_FLOP = re.compile(r"\*\*\* FLOP \*\*\* \[([^\]]+)\]")
RE_TURN = re.compile(r"\*\*\* TURN \*\*\* \[[^\]]+\] \[([^\]]+)\]")
RE_RIVER = re.compile(r"\*\*\* RIVER \*\*\* \[[^\]]+\] \[([^\]]+)\]")

RE_ACTION_GENERIC = re.compile(
    r"^(?P<player>[^:]+):\s+"
    r"(?P<action>folds|checks|calls|bets|raises|posts small blind|posts big blind)"
    r"(?:\s+\$?(?P<amt>[0-9]+(?:\.[0-9]+)?))?"
    r"(?:\s+to\s+\$?(?P<toamt>[0-9]+(?:\.[0-9]+)?))?"
    r"(?P<allin>.*all-in.*)?\s*$",
    re.IGNORECASE,
)

RE_UNCALLED = re.compile(
    r"^Uncalled bet \(\$([0-9.]+)\) returned to ([^ ]+.*)$")
RE_COLLECTED = re.compile(r"^([^:]+) collected \$([0-9.]+) from pot$")
RE_SHOWS = re.compile(r"^([^:]+): shows \[([^\]]+)\]")

STREET_MARKERS = {
    "*** HOLE CARDS ***": "PREFLOP",
    "*** FLOP ***": "FLOP",
    "*** TURN ***": "TURN",
    "*** RIVER ***": "RIVER",
}

RANK_MAP = {**{str(i): i for i in range(2, 10)}, "T": 10,
            "J": 11, "Q": 12, "K": 13, "A": 14}
SUIT_MAP = {"c": 0, "d": 1, "h": 2, "s": 3}


def split_hands(text: str) -> List[str]:
    hands = re.split(RE_HAND_SPLIT, text)
    return [h.strip() for h in hands if h.strip().startswith("PokerStars Hand #")]


def seat_to_position_6max(button_seat: int, hero_seat: int) -> str:
    dist = (hero_seat - button_seat) % 6
    return {0: "BTN", 1: "SB", 2: "BB", 3: "UTG", 4: "HJ", 5: "CO"}.get(dist, "UNK")


def parse_board(hand_text: str) -> List[str]:
    board = []
    m = RE_FLOP.search(hand_text)
    if m:
        board += m.group(1).split()
    m = RE_TURN.search(hand_text)
    if m:
        board += m.group(1).split()
    m = RE_RIVER.search(hand_text)
    if m:
        board += m.group(1).split()
    return board


def normalize_action(a: str) -> str:
    a = a.lower().strip()
    if a in ("posts small blind", "posts big blind"):
        return "post"
    if a.endswith("s"):
        a = a[:-1]
    return a


def card_to_rs(card: str) -> Tuple[int, int]:
    return (RANK_MAP[card[0]], SUIT_MAP[card[1]])


def straight_high(ranks: List[int]) -> Optional[int]:
    r = sorted(set(ranks), reverse=True)
    if 14 in r:
        r.append(1)
    for i in range(len(r) - 4):
        window = r[i:i + 5]
        if window[0] - window[4] == 4 and len(set(window)) == 5:
            return window[0] if window[0] != 1 else 5
    return None


def eval_7cards(cards: List[Tuple[int, int]]) -> Tuple:
    ranks = [c[0] for c in cards]
    suits = [c[1] for c in cards]

    cnt: Dict[int, int] = {}
    for r in ranks:
        cnt[r] = cnt.get(r, 0) + 1
    items = sorted(cnt.items(), key=lambda x: (x[1], x[0]), reverse=True)

    suit_cnt: Dict[int, int] = {}
    for s in suits:
        suit_cnt[s] = suit_cnt.get(s, 0) + 1
    flush_suit = None
    for s, c in suit_cnt.items():
        if c >= 5:
            flush_suit = s
            break

    if flush_suit is not None:
        flush_ranks = [r for (r, s) in cards if s == flush_suit]
        sf = straight_high(flush_ranks)
        if sf is not None:
            return (8, sf)

    if items[0][1] == 4:
        quad = items[0][0]
        kicker = max([r for r in ranks if r != quad])
        return (7, quad, kicker)

    trips = [r for r, c in items if c == 3]
    pairs = [r for r, c in items if c == 2]
    if trips:
        top_trip = max(trips)
        remaining = [r for r in trips if r != top_trip]
        if pairs or remaining:
            top_pair = max(pairs) if pairs else max(remaining)
            return (6, top_trip, top_pair)

    if flush_suit is not None:
        flush_cards = sorted(
            [r for (r, s) in cards if s == flush_suit], reverse=True)[:5]
        return (5, *flush_cards)

    st = straight_high(ranks)
    if st is not None:
        return (4, st)

    if trips:
        trip = max(trips)
        kickers = sorted([r for r in ranks if r != trip], reverse=True)[:2]
        return (3, trip, *kickers)

    if len(pairs) >= 2:
        top2 = sorted(pairs, reverse=True)[:2]
        kicker = max([r for r in ranks if r not in top2])
        return (2, top2[0], top2[1], kicker)

    if len(pairs) == 1:
        pr = pairs[0]
        kickers = sorted([r for r in ranks if r != pr], reverse=True)[:3]
        return (1, pr, *kickers)

    hi = sorted(ranks, reverse=True)[:5]
    return (0, *hi)


def equity_vs_one_opponent(hero_hole: List[str], opp_hole: List[str], board_known: List[str]) -> float:
    known = hero_hole + opp_hole + board_known
    known_set = set(known)

    deck = []
    for r in "23456789TJQKA":
        for s in "cdhs":
            c = f"{r}{s}"
            if c not in known_set:
                deck.append(c)

    missing = 5 - len(board_known)
    if missing < 0 or missing > 2:
        return float("nan")

    hero_rs = [card_to_rs(c) for c in hero_hole]
    opp_rs = [card_to_rs(c) for c in opp_hole]
    board_rs_known = [card_to_rs(c) for c in board_known]

    wins = ties = total = 0
    for add in itertools.combinations(deck, missing):
        total += 1
        board_rs = board_rs_known + [card_to_rs(c) for c in add]
        hero_rank = eval_7cards(hero_rs + board_rs)
        opp_rank = eval_7cards(opp_rs + board_rs)
        if hero_rank > opp_rank:
            wins += 1
        elif hero_rank == opp_rank:
            ties += 1

    return (wins + ties * 0.5) / total if total > 0 else float("nan")


def parse_one_hand(hand_text: str) -> Tuple[dict, List[dict]]:
    m = RE_HEADER.search(hand_text)
    if not m:
        raise ValueError("Failed to parse header.")
    hand_id = m.group("hand_id")
    sb = float(m.group("sb"))
    bb = float(m.group("bb"))

    m2 = RE_TABLE_BTN.search(hand_text)
    table = m2.group(1)
    button_seat = int(m2.group(2))

    seats: Dict[str, Tuple[int, float]] = {}
    for sm in RE_SEAT_STACK.finditer(hand_text):
        seat = int(sm.group(1))
        name = sm.group(2).strip()
        stack = float(sm.group(3))
        seats[name] = (seat, stack)

    hero_seat, hero_stack = seats[HERO]
    pos = seat_to_position_6max(button_seat, hero_seat)

    m4 = RE_DEALT.search(hand_text)
    hero_hole = m4.group(1).split() if m4 else []
    rake = float(RE_RAKE.search(hand_text).group(
        1)) if RE_RAKE.search(hand_text) else 0.0

    current_street = "PREFLOP"
    committed_street: Dict[str, Dict[str, float]] = {
        s: {} for s in ["PREFLOP", "FLOP", "TURN", "RIVER"]}
    put_in_street = {"PREFLOP": 0.0, "FLOP": 0.0, "TURN": 0.0, "RIVER": 0.0}
    returned_street = {"PREFLOP": 0.0, "FLOP": 0.0, "TURN": 0.0, "RIVER": 0.0}
    collected = 0.0

    pre_actions = []
    hero_open_limp = False
    hero_call_vs_raise = False
    hero_open_raise = False
    hero_bb_defend_call = False

    pot = 0.0
    allin_events: List[dict] = []
    pending_hero_allin = None
    board_now: List[str] = []
    shown: Dict[str, List[str]] = {}

    for raw in hand_text.splitlines():
        line = raw.strip()
        if not line:
            continue

        if line.startswith("*** FLOP ***"):
            current_street = "FLOP"
            m = RE_FLOP.search(line)
            if m:
                board_now = m.group(1).split()
            continue
        if line.startswith("*** TURN ***"):
            current_street = "TURN"
            m = RE_TURN.search(line)
            if m:
                board_now = (board_now[:3] if len(
                    board_now) >= 3 else board_now) + m.group(1).split()
            continue
        if line.startswith("*** RIVER ***"):
            current_street = "RIVER"
            m = RE_RIVER.search(line)
            if m:
                board_now = (board_now[:4] if len(
                    board_now) >= 4 else board_now) + m.group(1).split()
            continue
        if line.startswith("*** HOLE CARDS ***"):
            current_street = "PREFLOP"
            continue

        m_unc = RE_UNCALLED.match(line)
        if m_unc:
            amt = float(m_unc.group(1))
            who = m_unc.group(2).strip()
            pot -= amt
            if who == HERO:
                returned_street[current_street] += amt
            if who in committed_street[current_street]:
                committed_street[current_street][who] = max(
                    0.0, committed_street[current_street][who] - amt)
            continue

        m_col = RE_COLLECTED.match(line)
        if m_col:
            who = m_col.group(1).strip()
            amt = float(m_col.group(2))
            if who == HERO:
                collected += amt
            continue

        m_show = RE_SHOWS.match(line)
        if m_show:
            who = m_show.group(1).strip()
            cards = m_show.group(2).split()
            shown[who] = cards
            continue

        m_act = RE_ACTION_GENERIC.match(line)
        if not m_act:
            continue

        player = m_act.group("player").strip()
        action = normalize_action(m_act.group("action"))
        amt = float(m_act.group("amt")) if m_act.group("amt") else None
        toamt = float(m_act.group("toamt")) if m_act.group("toamt") else None
        is_allin = bool(m_act.group("allin"))

        if current_street == "PREFLOP" and action != "post":
            pre_actions.append((player, action, amt, toamt))

        if player not in committed_street[current_street]:
            committed_street[current_street][player] = 0.0

        added = 0.0
        if action == "post":
            if amt is not None:
                added = amt
                committed_street[current_street][player] += added
        elif action == "call":
            if amt is not None:
                added = amt
                committed_street[current_street][player] += added
        elif action == "bet":
            if amt is not None:
                added = amt
                committed_street[current_street][player] = amt
        elif action == "raise":
            if toamt is not None:
                already = committed_street[current_street][player]
                added = max(0.0, toamt - already)
                committed_street[current_street][player] = toamt

        pot += added

        if player == HERO and action in ("post", "call", "bet", "raise") and added > 0:
            put_in_street[current_street] += added

        if player == HERO and is_allin and action in ("bet", "raise", "call"):
            pending_hero_allin = {
                "hand_id": hand_id,
                "street": current_street,
                "board_known": " ".join(board_now),
                "pot_before": round(pot - added, 4),
                "hero_add": round(added, 4),
                "hero_hole": " ".join(hero_hole),
            }

        if pending_hero_allin and player != HERO and current_street == pending_hero_allin["street"]:
            if action == "call" and added > 0:
                pending_hero_allin["caller"] = player
                pending_hero_allin["opp_add"] = round(added, 4)
                pending_hero_allin["pot_after_call"] = round(pot, 4)
                allin_events.append(pending_hero_allin)
                pending_hero_allin = None
            elif action == "raise":
                pending_hero_allin = None

    hero_vpip = any(p == HERO and a in ("call", "raise")
                    for (p, a, _, __) in pre_actions)
    hero_pfr = any(p == HERO and a == "raise" for (p, a, _, __) in pre_actions)

    hero_threebet = False
    seen_raise = False
    for (p, a, _, _) in pre_actions:
        if a == "raise" and p != HERO:
            seen_raise = True
        if p == HERO and a == "raise" and seen_raise:
            hero_threebet = True
            break

    seen_raise = False
    hero_open_raise = False
    for (p, a, _, __) in pre_actions:
        if a == "raise" and p != HERO:
            seen_raise = True
        if p == HERO and a == "raise" and not seen_raise:
            hero_open_raise = True
            break

    hero_open_limp = False
    seen_raise = False
    for (p, a, _, __) in pre_actions:
        if a == "raise" and p != HERO:
            seen_raise = True
        if p == HERO and a == "call" and not seen_raise:
            hero_open_limp = True
            break

    hero_call_vs_raise = False
    seen_raise = False
    for (p, a, _, __) in pre_actions:
        if a == "raise" and p != HERO:
            seen_raise = True
        if p == HERO and a == "call" and seen_raise:
            hero_call_vs_raise = True
            break

    hero_bb_defend_call = pos == "BB" and hero_call_vs_raise

    hero_put_in = sum(put_in_street.values())
    hero_returned = sum(returned_street.values())
    hero_cost = hero_put_in - hero_returned
    hero_net = collected - hero_cost
    hero_net_bb = hero_net / bb

    allin_ev_rows = []
    for ev in allin_events:
        caller = ev.get("caller")
        if not caller:
            continue
        if HERO not in shown or caller not in shown:
            continue

        board_known = ev["board_known"].split() if ev["board_known"] else []
        if len(board_known) < 3:
            continue

        eq = equity_vs_one_opponent(
            hero_hole=shown[HERO],
            opp_hole=shown[caller],
            board_known=board_known,
        )
        if eq != eq:
            continue

        pot_before = ev["pot_before"]
        hero_add = ev["hero_add"]
        opp_add = ev.get("opp_add", 0.0)
        pot_after = ev.get("pot_after_call", pot_before + hero_add + opp_add)
        ev_inc = eq * pot_after - hero_add

        allin_ev_rows.append({
            "hand_id": hand_id,
            "street": ev["street"],
            "hero_hole": " ".join(shown[HERO]),
            "opp": caller,
            "opp_hole": " ".join(shown[caller]),
            "board_known": " ".join(board_known),
            "pot_before": pot_before,
            "hero_add": hero_add,
            "opp_add": opp_add,
            "pot_after_call": pot_after,
            "equity": round(eq, 6),
            "ev_increment_$": round(ev_inc, 4),
        })

    row = {
        "hand_id": hand_id,
        "sb": sb,
        "bb": bb,
        "table": table,
        "pos": pos,
        "hero_stack_start": hero_stack,
        "hole": " ".join(hero_hole),
        "rake": rake,
        "vpip": hero_vpip,
        "pfr": hero_pfr,
        "three_bet": hero_threebet,
        "open_raise": hero_open_raise,
        "open_limp": hero_open_limp,
        "call_vs_raise": hero_call_vs_raise,
        "bb_defend_call": hero_bb_defend_call,
        "putin_pre": round(put_in_street["PREFLOP"], 4),
        "putin_flop": round(put_in_street["FLOP"], 4),
        "putin_turn": round(put_in_street["TURN"], 4),
        "putin_river": round(put_in_street["RIVER"], 4),
        "returned_pre": round(returned_street["PREFLOP"], 4),
        "returned_flop": round(returned_street["FLOP"], 4),
        "returned_turn": round(returned_street["TURN"], 4),
        "returned_river": round(returned_street["RIVER"], 4),
        "collected": round(collected, 4),
        "hero_cost": round(hero_cost, 4),
        "hero_net": round(hero_net, 4),
        "hero_net_bb": round(hero_net_bb, 4),
        "board_final": " ".join(parse_board(hand_text)),
    }

    return row, allin_ev_rows


def save_equity_curve(df: pd.DataFrame, path: str = "equity_curve.png") -> None:
    if df.empty:
        return

    curve = df.copy().reset_index(drop=True)
    curve["hand_num"] = range(1, len(curve) + 1)
    curve["cum_net"] = curve["hero_net"].cumsum()
    curve["cum_net_bb"] = curve["hero_net_bb"].cumsum()

    plt.figure(figsize=(11, 6))
    plt.plot(curve["hand_num"], curve["cum_net_bb"],
             linewidth=1.8, label="Cumulative bb")
    plt.axhline(0, linewidth=1)
    plt.xlabel("Hands")
    plt.ylabel("Big blinds")
    plt.title(f"{HERO} Equity Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def main():
    text = Path(INPUT_FILE).read_text(encoding="utf-8", errors="ignore")
    hands = split_hands(text)
    print("Hands found:", len(hands))

    hand_rows = []
    allin_rows = []

    for h in hands:
        r, evs = parse_one_hand(h)
        hand_rows.append(r)
        allin_rows.extend(evs)

    df = pd.DataFrame(hand_rows)
    df.to_csv("hands_report.csv", index=False)

    if allin_rows:
        pd.DataFrame(allin_rows).to_csv("allin_ev.csv", index=False)
    else:
        pd.DataFrame(columns=[
            "hand_id", "street", "hero_hole", "opp", "opp_hole", "board_known",
            "pot_before", "hero_add", "opp_add", "pot_after_call", "equity", "ev_increment_$"
        ]).to_csv("allin_ev.csv", index=False)

    save_equity_curve(df, "equity_curve.png")

    vpip = df["vpip"].mean() * 100
    pfr = df["pfr"].mean() * 100
    tb = df["three_bet"].mean() * 100
    gap = vpip - pfr

    total_bb = df["hero_net_bb"].sum()
    bb100 = (total_bb / max(len(df), 1)) * 100
    rake100 = (df["rake"].sum() / max(len(df), 1)) * 100

    print(f"\nHERO: {HERO}")
    print(
        f"VPIP: {vpip:.1f}% | PFR: {pfr:.1f}% | 3bet: {tb:.1f}% | Gap: {gap:.1f}%")
    print(
        f"Total net: ${df['hero_net'].sum():.2f} ({total_bb:.1f} bb) | bb/100: {bb100:.1f}")
    print(f"Total rake: ${df['rake'].sum():.2f} | Rake/100: ${rake100:.2f}")

    print("\nBy position:")
    pos_stats = df.groupby("pos").agg(
        hands=("hand_id", "count"),
        vpip=("vpip", "mean"),
        pfr=("pfr", "mean"),
        threebet=("three_bet", "mean"),
        bb100=("hero_net_bb", lambda x: x.mean() * 100),
    )
    pos_stats[["vpip", "pfr", "threebet"]] = (
        pos_stats[["vpip", "pfr", "threebet"]] * 100).round(1)
    pos_stats["bb100"] = pos_stats["bb100"].round(1)
    print(pos_stats.sort_index().to_string())

    print("\nLeak counters:")
    print(
        f"Open-limps: {int(df['open_limp'].sum())}  ({df['open_limp'].mean() * 100:.1f}%)")
    print(
        f"Cold/flat calls vs raise: {int(df['call_vs_raise'].sum())}  ({df['call_vs_raise'].mean() * 100:.1f}%)")
    print(
        f"BB defend calls: {int(df['bb_defend_call'].sum())}  ({df['bb_defend_call'].mean() * 100:.1f}%)")

    print("\n10 biggest losing hands (net bb):")
    worst = df.sort_values("hero_net_bb").head(
        10)[["hand_id", "pos", "hole", "hero_net_bb", "hero_net", "board_final"]]
    print(worst.to_string(index=False))

    evdf = pd.DataFrame(allin_rows)
    if not evdf.empty:
        print("\nAll-in EV (heads-up, shown cards only):")
        print(
            f"All-in spots: {len(evdf)} | Avg equity: {evdf['equity'].mean():.3f} | Sum EV(increment): ${evdf['ev_increment_$'].sum():.2f}"
        )
    else:
        print("\nAll-in EV: no eligible hands.")

    print("\nWrote: hands_report.csv, allin_ev.csv, equity_curve.png")


if __name__ == "__main__":
    main()

