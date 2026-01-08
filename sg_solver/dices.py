"""
Star Genius dice definitions.

The game has 7 dice:
- 3x 8-faced dice
- 4x 6-faced dice (2 incomplete in data)
"""

import random

# 8-faced dice (3 of them)
DICE_8_1 = [25, 26, 36, 37, 38, 40, 45, 47]
DICE_8_2 = [2, 4, 7, 8, 9, 17, 11, 16]
DICE_8_3 = [12, 13, 23, 24, 32, 33, 41, 42]

# 6-faced dice (4 of them, 2 incomplete)
DICE_6_1 = [1, 5, 15, 34, 44, 48]
DICE_6_2 = [19, 20, 21, 28, 29, 30]
DICE_6_3 = [18, 22, 39]  # incomplete - only 3 faces known
DICE_6_4 = [10, 27, 31]  # incomplete - only 3 faces known

ALL_DICE = [DICE_8_1, DICE_8_2, DICE_8_3, DICE_6_1, DICE_6_2, DICE_6_3, DICE_6_4]


def roll_dice() -> list[int]:
    """Roll all 7 dice, return 7 cell IDs to block."""
    return [random.choice(die) for die in ALL_DICE]


def fixed_roll() -> list[int]:
    """A fixed roll for testing (first face of each die)."""
    return [die[0] for die in ALL_DICE]


# Some specific test cases
TEST_ROLLS = [
    # First faces
    [25, 2, 12, 1, 19, 18, 10],
    # Mix of faces
    [36, 7, 23, 5, 20, 22, 27],
    # Another mix
    [45, 16, 42, 48, 30, 39, 31],
]


if __name__ == "__main__":
    print("Fixed roll:", fixed_roll())
    print("Random roll:", roll_dice())
    print()
    print("Test rolls:")
    for i, roll in enumerate(TEST_ROLLS):
        print(f"  Test {i+1}: {roll}")
