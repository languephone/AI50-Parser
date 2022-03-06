import nltk
import sys

TERMINALS = """
Adj -> "country" | "dreadful" | "enigmatical" | "little" | "moist" | "red"
Adv -> "down" | "here" | "never"
Conj -> "and" | "until"
Det -> "a" | "an" | "his" | "my" | "the"
N -> "armchair" | "companion" | "day" | "door" | "hand" | "he" | "himself"
N -> "holmes" | "home" | "i" | "mess" | "paint" | "palm" | "pipe" | "she"
N -> "smile" | "thursday" | "walk" | "we" | "word"
P -> "at" | "before" | "in" | "of" | "on" | "to"
V -> "arrived" | "came" | "chuckled" | "had" | "lit" | "said" | "sat"
V -> "smiled" | "tell" | "were"
"""

NONTERMINALS = """
S -> NP VP
VP -> V | V NP | VP Conj VP | NP VP | Adv VP
NP -> N | Det NP | N NP | P NP | Adj N | N Adj | NP Adv
"""

grammar = nltk.CFG.fromstring(NONTERMINALS + TERMINALS)
parser = nltk.ChartParser(grammar)


def main():

    # If filename specified, read sentence from file
    if len(sys.argv) == 2:
        with open(sys.argv[1]) as f:
            s = f.read()

    # Otherwise, get sentence as input
    else:
        s = input("Sentence: ")

    # Convert input into list of words
    s = preprocess(s)

    # Attempt to parse sentence
    try:
        trees = list(parser.parse(s))
    except ValueError as e:
        print(e)
        return
    if not trees:
        print("Could not parse sentence.")
        return

    # Print each tree with noun phrase chunks
    for tree in trees:
        tree.pretty_print()

        print("Noun Phrase Chunks")
        for np in np_chunk(tree):
            print(" ".join(np.flatten()))


def preprocess(sentence):
    """
    Convert `sentence` to a list of its words.
    Pre-process sentence by converting all characters to lowercase
    and removing any word that does not contain at least one alphabetic
    character.
    """
    # Use nltk's tokenizer to convert sentence into words
    tokens = nltk.word_tokenize(sentence)
    processed_tokens = []
    # Convert words to lowercase & include only alphanumeric tokens
    for token in tokens:
        # Assume that the token doesn't have any alphabetic characters
        alpha = False
        # If an alphabetic character is found, change alpha to True
        for character in token:
            if character.isalpha():
                alpha = True
                break
        if alpha:
            processed_tokens.append(token.lower())

    return processed_tokens


def np_chunk(tree):
    """
    Return a list of all noun phrase chunks in the sentence tree.
    A noun phrase chunk is defined as any subtree of the sentence
    whose label is "NP" that does not itself contain any other
    noun phrases as subtrees.
    """
    
    np_chunks = []

    # Iterate through subtrees labelled as 'NP'
    for subtree in tree.subtrees(lambda x: x.label() == 'NP'):
        # Check if the 'NP' subtree has child trees labelled as 'NP'
        labels = [child.label() for child in subtree.subtrees()]
        # If only one 'NP' label, then there are no sub-sub-NP trees
        if labels.count('NP') == 1:
            np_chunks.append(subtree)

    return np_chunks


if __name__ == "__main__":
    main()
