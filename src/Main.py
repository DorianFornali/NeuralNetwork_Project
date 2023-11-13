import pandas as pd

### on lit les databases

ranking = pd.read_csv("../databases/fifa_ranking-2023-07-20.csv")
goalscorers = pd.read_csv("../databases/goalscorers.csv")
results= pd.read_csv("../databases/results.csv")
shootouts= pd.read_csv("../databases/shootouts.csv")

##On déclare la date de début ainsi que la date de fin
start_date = '1996-01-01'
end_date = "2022-12-18"

#Ici la date de la database ranking s'appelle rank_date. On la remplace avec date pour homogéniser avec les autres
ranking.rename(columns={'rank_date': 'date'}, inplace=True)

#On filtre par date les databases
rankings_filtered = ranking[(ranking['date'] >= start_date) & (ranking['date'] <= end_date)]
goalscorers_filtered = goalscorers[(goalscorers['date'] >= start_date) & (goalscorers['date'] <= end_date)]
results_filtered = results[(results['date'] >= start_date) & (results['date'] <= end_date)]
shoothouts_filtered = shootouts[(shootouts['date'] >= start_date) & (shootouts['date'] <= end_date)]

#On prend que les matchs de Coupe du Monde ou de Qualification à la coupe du Monde
results_filtered = results_filtered[results_filtered['tournament'].isin(['FIFA World Cup', 'FIFA World Cup qualification'])]

#On met un peu d'ordre dans la data base rankings, on ordonne premièrement par date pui par classement
rankings_filtered = rankings_filtered.sort_values(by=['date', 'rank'])

#On crée des nouvelles databases avec les nouvelles modifications et filtres apportés
rankings_filtered.to_csv('rankings_filtered.csv', index=False)
goalscorers_filtered.to_csv('goalscorers_filtered.csv', index=False)
results_filtered.to_csv('results_filtered.csv', index=False)
shoothouts_filtered.to_csv('shoothouts_filtered.csv', index=False)

#Conseil: faites des prints de temps en temps pour pouvoir voir l'état des databases

if __name__ == '__main__':
    results = pd.read_csv("../databases/results.csv")
    ranking = pd.read_csv("../databases/fifa_ranking-2023-07-20.csv")

    print(ranking)

