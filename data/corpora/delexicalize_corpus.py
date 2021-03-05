import pyconll

from sys import argv

lex_rels = {"obl", "nmod", "advcl", "acl", "conj"}

CASES = {"nom", "gen", "dat", "acc", "voc", "loc", "ins"}

def delexicalise(corpus_filename):
    raw_sents = pyconll.load_from_file(corpus_filename)

    for sent in raw_sents:
        for token in sent:
            for (head_id, (rel_type, rel_subtype1, rel_subtype2, rel_subtype3)) in token.deps.items():
                token.deps[head_id] = delex_relation(rel_type, rel_subtype1, rel_subtype2, rel_subtype3)

        print(sent.conll())
        print()


def delex_relation(rel_type, rel_subtype1, rel_subtype2, rel_subtype3):
    if (rel_subtype1, rel_subtype2, rel_subtype3) == (None, None, None):  # Nothing to lexicalize
        return (rel_type, rel_subtype1, rel_subtype2, rel_subtype3)

    if rel_type == "obl":
        if rel_subtype1 in {"tmod", "npmod"}:
            assert (rel_subtype2, rel_subtype3) == (None, None)
            return (rel_type, rel_subtype1, rel_subtype2, rel_subtype3)
        elif rel_subtype1 == "agent":
            if rel_subtype2 is not None:
                return (rel_type, rel_subtype1, "[case]", rel_subtype3)
            else:
                return (rel_type, rel_subtype1, rel_subtype2, rel_subtype3)
        elif rel_subtype1 in CASES:
            assert (rel_subtype2, rel_subtype3) == (None, None)
            return (rel_type, rel_subtype1, rel_subtype2, rel_subtype3)
        elif rel_subtype1 == "arg":
            if rel_subtype2 in CASES:
                assert rel_subtype3 is None
                return (rel_type, rel_subtype1, rel_subtype2, rel_subtype3)
            else:  # Subtype 2 is lexical material
                assert rel_subtype3 in CASES or rel_subtype3 is None
                return (rel_type, rel_subtype1, "[case]", rel_subtype3)
        else:  # Subtype 1 is lexical material
            assert rel_subtype2 in CASES or rel_subtype2 is None
            assert rel_subtype3 is None
            return (rel_type, "[case]", rel_subtype2, rel_subtype3)


    elif rel_type == "nmod":
        if rel_subtype1 in {"tmod", "npmod", "poss"} and (rel_subtype2, rel_subtype3) == (None, None):
            return (rel_type, rel_subtype1, rel_subtype2, rel_subtype3)
        elif rel_subtype1 in CASES:
            assert (rel_subtype2, rel_subtype3) == (None, None)
            return (rel_type, rel_subtype1, rel_subtype2, rel_subtype3)
        else:
            assert rel_subtype2 in CASES or rel_subtype2 is None
            assert rel_subtype3 is None
            return (rel_type, "[case]", rel_subtype2, rel_subtype3)

    elif rel_type == "advcl":
        return (rel_type, "[mark]", rel_subtype2, rel_subtype3)

    elif rel_type == "acl":
        if rel_subtype1 == "relcl" or rel_subtype1 in CASES:
            rel_subtype2 = None
            assert rel_subtype3 == None
            return (rel_type, rel_subtype1, rel_subtype2, rel_subtype3)
        else:
            return (rel_type, "[mark]", rel_subtype2, rel_subtype3)

    elif rel_type == "conj":
        return (rel_type, "[cc]", rel_subtype2, rel_subtype3)

    else:
        return (rel_type, rel_subtype1, rel_subtype2, rel_subtype3)


if __name__ == "__main__":
    corpus_filename = argv[1]
    delexicalise(corpus_filename)
