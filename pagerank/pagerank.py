import os
import random
import re
import sys

DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """
    # If no outgoing links, treat as linking to all pages
    links = corpus.get(page)
    if not links:
        links = set(corpus.keys())
    n_pages = len(corpus)
    prob = {}
    # Base probability for random jump
    base_prob = (1 - damping_factor) / n_pages
    # Distribute damping probability among linked pages
    for p in corpus:
        prob[p] = base_prob
    for linked in links:
        prob[linked] += damping_factor / len(links)
    return prob


def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    pages = list(corpus.keys())
    counts = {p: 0 for p in pages}
    # First sample: random page
    current = random.choice(pages)
    counts[current] += 1
    # Remaining samples
    for _ in range(1, n):
        distribution = transition_model(corpus, current, damping_factor)
        # Choose next based on distribution
        next_page = random.choices(
            population=list(distribution.keys()),
            weights=list(distribution.values()),
            k=1
        )[0]
        counts[next_page] += 1
        current = next_page
    # Convert counts to probabilities
    return {p: counts[p] / n for p in pages}


def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    pages = list(corpus.keys())
    n_pages = len(pages)
    # Initialize ranks
    ranks = {p: 1 / n_pages for p in pages}
    # Precompute incoming links (including pages with no outgoing treated specially)
    incoming = {p: set() for p in pages}
    for p in pages:
        if corpus[p]:
            for q in corpus[p]:
                incoming[q].add(p)
        else:
            # p has no links: treat as linking to all pages
            for q in pages:
                incoming[q].add(p)
    # Iterate until convergence
    while True:
        new_ranks = {}
        for p in pages:
            # Random jump factor
            rank = (1 - damping_factor) / n_pages
            # Summation over incoming links
            summation = 0
            for q in incoming[p]:
                # Outgoing count for q
                links_q = corpus[q] if corpus[q] else set(pages)
                summation += ranks[q] / len(links_q)
            rank += damping_factor * summation
            new_ranks[p] = rank
        # Check convergence
        if all(abs(new_ranks[p] - ranks[p]) < 0.001 for p in pages):
            break
        ranks = new_ranks
    return ranks


if __name__ == "__main__":
    main()
