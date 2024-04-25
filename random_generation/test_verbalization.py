from verb.verbalize import IndependentStatementVerbalization


# if __name__ == '__main__':
def test_verbalization():
    verbalizer = IndependentStatementVerbalization(None)
    out = verbalizer.fl_2_nl('x = circle x A B C')
    print()
    print(out)