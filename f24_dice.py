#!/usr/bin/env python3
"""F24 (optional, reviewer): natural-language dice. A game state with one number C, then a request
to roll a die for the player; q(roll) is read after the prefilled `roll=`. C takes every face value
of the die plus 3 out-of-range digits. No other digits appear in the prompt."""
import json
from pathlib import Path

STATES = [
    ("hp", "The player has {c} HP left."),
    ("floor", "The player is on floor {c} of the dungeon."),
    ("potions", "The player carries {c} healing potions."),
    ("turn", "It is turn {c} of the battle."),
    ("keys", "The player has found {c} keys so far."),
    ("enemies", "There are {c} goblins in the room."),
    ("gold", "The player's purse holds {c} gold coins."),
    ("level", "The player's character is level {c}."),
    ("arrows", "The player has {c} arrows in the quiver."),
    ("lives", "The player has {c} lives remaining."),
    ("day", "It is day {c} of the journey."),
    ("room", "The party has just entered room {c}."),
]
DICE = {"d6": list(range(1, 7)), "d4": list(range(1, 5))}


def oor(legal):
    lo, hi = min(legal), max(legal)
    c = [d for d in range(hi + 1, 10)] + [d for d in range(lo - 1, -1, -1)]
    return [d for d in c if d not in legal][:3]


def text(state, die, c):
    return ("You are the game master of a text role-playing game.\n" + state.format(c=c) + "\n"
            f"The player attacks. Roll a {die} for the attack now.\n"
            "Reply with exactly one line in this format:\nroll=<digit>\nDo not explain or add any other text.")


def items():
    out = []
    for key, st in STATES:
        for die, legal in DICE.items():
            for c in legal + oor(legal):
                out.append({"iid": f"f24|{key}|{die}|{c}", "sid": f"{key}|{die}", "state": key, "die": die,
                            "legal": legal, "oor": oor(legal), "c": c, "c_in_range": c in legal,
                            "prefix": "roll=", "text": text(st, die, c)})
    return out


if __name__ == "__main__":
    its = items()
    Path("f24_items.jsonl").write_text("\n".join(json.dumps(i) for i in its) + "\n")
    print(len(its), "items"); print(its[0]["text"])
