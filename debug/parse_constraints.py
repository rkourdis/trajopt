from pprint import pprint

if __name__ == "__main__":
    constraints = {}

    with open("constraints.txt", "r") as rf:
        for line in rf:
            index = int(line[2:line.index(']')])
            viol = float(line[line.index('=')+1 : line.index(',')])

            constraints[index] = viol

    print("Top 10 unsatisfied constraints:")
    pprint(sorted(constraints.items(), key = lambda item: item[1])[:10])
    