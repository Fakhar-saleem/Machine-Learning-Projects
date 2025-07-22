import sys

from crossword import *


class CrosswordCreator():

    def __init__(self, crossword):
        """
        Create new CSP crossword generate.
        """
        self.crossword = crossword
        self.domains = {
            var: set(self.crossword.words)
            for var in self.crossword.variables
        }

    def letter_grid(self, assignment):
        """
        Return 2D array representing a given assignment.
        """
        letters = [
            [None for _ in range(self.crossword.width)]
            for _ in range(self.crossword.height)
        ]
        for variable, word in assignment.items():
            direction = variable.direction
            for k in range(len(word)):
                i = variable.i + (k if direction == Variable.DOWN else 0)
                j = variable.j + (k if direction == Variable.ACROSS else 0)
                letters[i][j] = word[k]
        return letters

    def print(self, assignment):
        """
        Print crossword assignment to the terminal.
        """
        letters = self.letter_grid(assignment)
        for i in range(self.crossword.height):
            for j in range(self.crossword.width):
                if self.crossword.structure[i][j]:
                    print(letters[i][j] or " ", end="")
                else:
                    print("█", end="")
            print()

    def solve(self):
        """
        Enforce node and arc consistency, and then solve via backtracking.
        """
        self.enforce_node_consistency()
        self.ac3()
        return self.backtrack({})

    def enforce_node_consistency(self):
        """
        Remove any values that don't match variable length.
        """
        for var in self.domains:
            to_remove = {word for word in self.domains[var] if len(word) != var.length}
            self.domains[var] -= to_remove

    def revise(self, x, y):
        """
        Make x arc consistent with y.
        Remove values from self.domains[x] lacking a match in y.
        """
        revised = False
        overlap = self.crossword.overlaps[x, y]
        if overlap is None:
            return False
        i, j = overlap
        remove = set()
        for wx in self.domains[x]:
            # any wy in y domain with matching char at overlap?
            if not any(wy[j] == wx[i] for wy in self.domains[y]):
                remove.add(wx)
        if remove:
            self.domains[x] -= remove
            revised = True
        return revised

    def ac3(self, arcs=None):
        """
        Enforce arc consistency over all arcs or given list.
        """
        queue = []
        if arcs:
            queue = list(arcs)
        else:
            for x in self.domains:
                for y in self.crossword.neighbors(x):
                    queue.append((x, y))
        while queue:
            x, y = queue.pop(0)
            if self.revise(x, y):
                if not self.domains[x]:
                    return False
                for z in self.crossword.neighbors(x) - {y}:
                    queue.append((z, x))
        return True

    def assignment_complete(self, assignment):
        """
        Check if all variables assigned.
        """
        return set(assignment.keys()) == self.crossword.variables

    def consistent(self, assignment):
        """
        Check if assignment is consistent:
          - all values distinct
          - lengths correct
          - overlaps match
        """
        # distinct
        if len(set(assignment.values())) < len(assignment):
            return False
        # each var length
        for var, word in assignment.items():
            if len(word) != var.length:
                return False
        # overlaps
        for v1, w1 in assignment.items():
            for v2, w2 in assignment.items():
                if v1 == v2:
                    continue
                overlap = self.crossword.overlaps[v1, v2]
                if overlap:
                    i, j = overlap
                    if w1[i] != w2[j]:
                        return False
        return True

    def order_domain_values(self, var, assignment):
        """
        Return domain values ordered by least-constraining heuristic.
        """
        def count_conflicts(word):
            count = 0
            for nbr in self.crossword.neighbors(var):
                if nbr not in assignment:
                    overlap = self.crossword.overlaps[var, nbr]
                    if overlap:
                        i, j = overlap
                        for w2 in self.domains[nbr]:
                            if w2[j] != word[i]:
                                count += 1
            return count
        return sorted(self.domains[var], key=count_conflicts)

    def select_unassigned_variable(self, assignment):
        """
        Select unassigned var by MRV, break ties by degree.
        """
        unassigned = [v for v in self.domains if v not in assignment]
        # MRV
        unassigned.sort(key=lambda v: (len(self.domains[v]), -len(self.crossword.neighbors(v))))
        return unassigned[0] if unassigned else None

    def backtrack(self, assignment):
        """
        Backtracking search for a solution. Return full assignment or None.
        """
        if self.assignment_complete(assignment):
            return assignment
        var = self.select_unassigned_variable(assignment)
        for value in self.order_domain_values(var, assignment):
            new_assign = assignment.copy()
            new_assign[var] = value
            if self.consistent(new_assign):
                result = self.backtrack(new_assign)
                if result:
                    return result
        return None


def main():
    # Check usage
    if len(sys.argv) not in [3, 4]:
        sys.exit("Usage: python generate.py structure words [output]")
    structure = sys.argv[1]
    words = sys.argv[2]
    output = sys.argv[3] if len(sys.argv) == 4 else None
    crossword = Crossword(structure, words)
    creator = CrosswordCreator(crossword)
    assignment = creator.solve()
    if assignment is None:
        print("No solution.")
    else:
        creator.print(assignment)
        if output:
            creator.save(assignment, output)


if __name__ == "__main__":
    main()
