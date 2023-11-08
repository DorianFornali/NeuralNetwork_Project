import pandas as pd

if __name__ == '__main__':
    results = pd.read_csv("../databases/results.csv")
    ranking = pd.read_csv("../databases/fifa_ranking-2023-07-20.csv")

    print(ranking)

