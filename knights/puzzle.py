from logic import *

# Symbols for whether each character is a knight or a knave
AKnight = Symbol("A is a Knight")
AKnave = Symbol("A is a Knave")

BKnight = Symbol("B is a Knight")
BKnave = Symbol("B is a Knave")

CKnight = Symbol("C is a Knight")
CKnave = Symbol("C is a Knave")

# Symbols for what A actually said in Puzzle 3
A_said_knight = Symbol("A said 'I am a knight'")
A_said_knave = Symbol("A said 'I am a knave'")

# Puzzle 0
# A says "I am both a knight and a knave."
knowledge0 = And(
    # A is either a knight or a knave, but not both
    Or(AKnight, AKnave),
    Not(And(AKnight, AKnave)),

    # If A is a knight, then his statement is true; if a knave, false
    Implication(AKnight, And(AKnight, AKnave)),
    Implication(AKnave, Not(And(AKnight, AKnave)))
)

# Puzzle 1
# A says "We are both knaves."
knowledge1 = And(
    # Two characters
    Or(AKnight, AKnave), Not(And(AKnight, AKnave)),
    Or(BKnight, BKnave), Not(And(BKnight, BKnave)),

    # A's statement
    Implication(AKnight, And(AKnave, BKnave)),
    Implication(AKnave, Not(And(AKnave, BKnave)))
)

# Puzzle 2
# A says "We are the same kind."  B says "We are of different kinds."
knowledge2 = And(
    # Two characters
    Or(AKnight, AKnave), Not(And(AKnight, AKnave)),
    Or(BKnight, BKnave), Not(And(BKnight, BKnave)),

    # A's statement
    Implication(AKnight, Or(And(AKnight, BKnight), And(AKnave, BKnave))),
    Implication(AKnave, Not(Or(And(AKnight, BKnight), And(AKnave, BKnave)))),

    # B's statement
    Implication(BKnight, Or(And(AKnight, BKnave), And(AKnave, BKnight))),
    Implication(BKnave, Not(Or(And(AKnight, BKnave), And(AKnave, BKnight))))
)

# Puzzle 3
# A says either "I am a knight." or "I am a knave."  (unknown which)
# B says "A said 'I am a knave'."  and  "C is a knave."
# C says "A is a knight."
knowledge3 = And(
    # Each person is exactly one of knight or knave
    Or(AKnight, AKnave), Not(And(AKnight, AKnave)),
    Or(BKnight, BKnave), Not(And(BKnight, BKnave)),
    Or(CKnight, CKnave), Not(And(CKnight, CKnave)),

    # A's utterance: exactly one statement
    Or(A_said_knight, A_said_knave),
    Implication(A_said_knight, Not(A_said_knave)),
    Implication(A_said_knave, Not(A_said_knight)),

    # If A is a knight, his utterance must be true; if a knave, false
    Implication(AKnight, Biconditional(A_said_knight, AKnight)),
    Implication(AKnight, Biconditional(A_said_knave, AKnave)),
    Implication(AKnave, Not(A_said_knight)),
    Implication(AKnave, Not(A_said_knave)),

    # B's statements: links his truthfulness to A's utterance and C's nature
    Biconditional(BKnight, A_said_knave),
    Biconditional(BKnight, CKnave),

    # C's statement
    Biconditional(CKnight, AKnight)
)


def main():
    symbols = [AKnight, AKnave, BKnight, BKnave, CKnight, CKnave]
    puzzles = [
        ("Puzzle 0", knowledge0),
        ("Puzzle 1", knowledge1),
        ("Puzzle 2", knowledge2),
        ("Puzzle 3", knowledge3)
    ]
    for puzzle, knowledge in puzzles:
        print(puzzle)
        for symbol in symbols:
            if model_check(knowledge, symbol):
                print(f"    {symbol}")

if __name__ == "__main__":
    main()
