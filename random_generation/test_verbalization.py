from verb.verbalize import IndependentStatementVerbalization


def test_verbalization():
    verbalizer = IndependentStatementVerbalization(None)

    problem = 'A B C = triangle A B C; X = circle X A B C'
    problem = 'X = circle X A B C'

    out = verbalizer.problem_fl_2_nl(problem)
    print()
    print(out)

    for _ in range(10):
        print(verbalizer.problem_fl_2_nl(problem))


if __name__ == '__main__':
    test_verbalization()
