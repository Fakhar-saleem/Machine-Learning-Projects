import csv
import itertools
import sys

PROBS = {

    # Unconditional probabilities for having gene
    "gene": {
        2: 0.01,
        1: 0.03,
        0: 0.96
    },

    "trait": {

        # Probability of trait given two copies of gene
        2: {
            True: 0.65,
            False: 0.35
        },

        # Probability of trait given one copy of gene
        1: {
            True: 0.56,
            False: 0.44
        },

        # Probability of trait given no gene
        0: {
            True: 0.01,
            False: 0.99
        }
    },

    # Mutation probability
    "mutation": 0.01
}


def main():

    # Check for proper usage
    if len(sys.argv) != 2:
        sys.exit("Usage: python heredity.py data.csv")
    people = load_data(sys.argv[1])

    # Keep track of gene and trait probabilities for each person
    probabilities = {
        person: {
            "gene": {2: 0, 1: 0, 0: 0},
            "trait": {True: 0, False: 0}
        }
        for person in people
    }

    # Loop over all sets of people who might have the trait
    names = set(people)
    for have_trait in powerset(names):

        # Check if current set of people violates known information
        fails_evidence = any(
            (people[person]["trait"] is not None and
             people[person]["trait"] != (person in have_trait))
            for person in names
        )
        if fails_evidence:
            continue

        # Loop over all sets of people who might have the gene
        for one_gene in powerset(names):
            for two_genes in powerset(names - one_gene):

                # Update probabilities with new joint probability
                p = joint_probability(people, one_gene, two_genes, have_trait)
                update(probabilities, one_gene, two_genes, have_trait, p)

    # Ensure probabilities sum to 1
    normalize(probabilities)

    # Print results
    for person in people:
        print(f"{person}:")
        for field in probabilities[person]:
            print(f"  {field.capitalize()}:")
            for value in probabilities[person][field]:
                p = probabilities[person][field][value]
                print(f"    {value}: {p:.4f}")


def load_data(filename):
    """
    Load gene and trait data from a file into a dictionary.
    File assumed to be a CSV containing fields name, mother, father, trait.
    mother, father must both be blank, or both be valid names in the CSV.
    trait should be 0 or 1 if trait is known, blank otherwise.
    """
    data = dict()
    with open(filename) as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["name"]
            data[name] = {
                "name": name,
                "mother": row["mother"] or None,
                "father": row["father"] or None,
                "trait": (True if row["trait"] == "1" else
                          False if row["trait"] == "0" else None)
            }
    return data


def powerset(s):
    """
    Return a list of all possible subsets of set s.
    """
    s = list(s)
    return [
        set(combo) for combo in itertools.chain.from_iterable(
            itertools.combinations(s, r) for r in range(len(s) + 1)
        )
    ]


def joint_probability(people, one_gene, two_genes, have_trait):
    """
    Compute and return a joint probability.
    """
    probability = 1.0

    # Helper to compute probability parent passes gene
    def pass_prob(parent):
        if parent in two_genes:
            return 1 - PROBS["mutation"]
        elif parent in one_gene:
            return 0.5
        else:
            return PROBS["mutation"]

    for person in people:
        # Determine gene count
        if person in two_genes:
            genes = 2
        elif person in one_gene:
            genes = 1
        else:
            genes = 0

        # Compute gene probability
        mother = people[person]["mother"]
        father = people[person]["father"]
        if mother is None and father is None:
            gene_prob = PROBS["gene"][genes]
        else:
            pm = pass_prob(mother)
            pf = pass_prob(father)
            if genes == 2:
                gene_prob = pm * pf
            elif genes == 1:
                gene_prob = pm * (1 - pf) + (1 - pm) * pf
            else:
                gene_prob = (1 - pm) * (1 - pf)

        # Compute trait probability
        has_trait = person in have_trait
        trait_prob = PROBS["trait"][genes][has_trait]

        # Multiply into joint probability
        probability *= gene_prob * trait_prob

    return probability


def update(probabilities, one_gene, two_genes, have_trait, p):
    """
    Add to `probabilities` a new joint probability `p`.
    """
    for person in probabilities:
        # Determine gene count
        if person in two_genes:
            genes = 2
        elif person in one_gene:
            genes = 1
        else:
            genes = 0
        # Update gene distribution
        probabilities[person]["gene"][genes] += p
        # Update trait distribution
        has_trait = person in have_trait
        probabilities[person]["trait"][has_trait] += p


def normalize(probabilities):
    """
    Update `probabilities` such that each probability distribution is normalized.
    """
    for person in probabilities:
        # Normalize gene distribution
        gene_dist = probabilities[person]["gene"]
        total_genes = sum(gene_dist.values())
        if total_genes > 0:
            for g in gene_dist:
                gene_dist[g] /= total_genes
        # Normalize trait distribution
        trait_dist = probabilities[person]["trait"]
        total_traits = sum(trait_dist.values())
        if total_traits > 0:
            for t in trait_dist:
                trait_dist[t] /= total_traits


if __name__ == "__main__":
    main()
