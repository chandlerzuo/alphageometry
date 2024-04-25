from verb.verbalize import IndependentStatementVerbalization


def test_verbalization():
    verbalizer = IndependentStatementVerbalization(None)
    out = verbalizer.fl_2_nl('x = circle x A B C')
    print()
    print(out)


if __name__ == '__main__':
    test_verbalization()
