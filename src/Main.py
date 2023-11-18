import pandas as pd

def to_CSV(dataframe, nomDataframe, booleanIndex):
    path = f'../outputs/{nomDataframe}'
    dataframe.to_csv(path, index=booleanIndex)

import pandas as pd

if __name__ == '__main__':

    ### on lit les databases

    ranking = pd.read_csv("../databases/fifa_ranking-2023-07-20.csv")
    goalscorers = pd.read_csv("../databases/goalscorers.csv")
    results = pd.read_csv("../databases/results.csv")
    shootouts = pd.read_csv("../databases/shootouts.csv")

    ## On déclare la date de début ainsi que la date de fin

    start_date = '1996-01-01'
    end_date = "2022-12-18"

    # Ici la date de la database ranking s'appelle rank_date. On la remplace avec date pour homogéniser avec les autres
    ranking.rename(columns={'rank_date': 'date'}, inplace=True)

    # On filtre par date les databases
    rankings_filtered = ranking[(ranking['date'] >= start_date) & (ranking['date'] <= end_date)]
    goalscorers_filtered = goalscorers[(goalscorers['date'] >= start_date) & (goalscorers['date'] <= end_date)]
    results_filtered = results[(results['date'] >= start_date) & (results['date'] <= end_date)]
    shoothouts_filtered = shootouts[(shootouts['date'] >= start_date) & (shootouts['date'] <= end_date)]

    # On prend que les matchs de Coupe du Monde ou de Qualification à la coupe du Monde
    results_filtered = results_filtered[results_filtered['tournament'].isin(['FIFA World Cup', 'FIFA World Cup qualification'])]

    # On met un peu d'ordre dans la data base rankings, on ordonne premièrement par date pui par classement
    rankings_filtered = rankings_filtered.sort_values(by=['date', 'rank'])



    # On convertit les dates en datetime, pratique pour la suite
    results_filtered['date'] = pd.to_datetime(results_filtered['date'])
    rankings_filtered['date'] = pd.to_datetime(rankings_filtered['date'])

    # On crée daily_ranking qui est une version de rankings_filtered avec un classement par jour
    daily_ranking = rankings_filtered.set_index('date')
    daily_ranking = daily_ranking.groupby('country_full').resample('D').first()
    daily_ranking.ffill(inplace=True)

    # On écrit nos dataframe en csv dans le dossier outputs
    to_CSV(rankings_filtered, "rankings_filtered", False)
    to_CSV(goalscorers_filtered, "goalscorers_filtered", False)
    to_CSV(results_filtered, "results_filtered", False)
    to_CSV(shoothouts_filtered, "shoothouts_filtered", False)

    #to_CSV(daily_ranking, "daily_ranking", True)# Ne pas le mettre dans github, le fichier est trop gros

    #2.3 merging Data
    #we will build new dataframe named rera

    dropped_daily_ranking= daily_ranking.drop('country_full', axis=1) #I dropped country_full because there are 2 columns named country_full in daily_ranking
    dropped_daily_ranking=dropped_daily_ranking.reset_index()#date and country_full are now normal columns not indexes anymore


    rera = pd.merge(results_filtered, dropped_daily_ranking, left_on=['home_team', 'date'], right_on=['country_full', 'date'], how='left')

    # Renaming columns  for the home team
    rera = rera.rename(columns={
        'rank': 'home_rank',
        'total_points': 'home_total_points',
        'previous_points': 'home_previous_points',
        'rank_change': 'home_rank_change'
    })

    #Merging the datasets based on away_team and date
    rera = pd.merge(rera, dropped_daily_ranking, left_on=['away_team', 'date'], right_on=['country_full', 'date'],
                         how='left')

    # Renaming columns  for the away team
    rera = rera.rename(columns={
        'rank': 'away_rank',
        'total_points': 'away_total_points',
        'previous_points': 'away_previous_points',
        'rank_change': 'away_rank_change'
    })

# question 4 (did question 4 before 3 to clean rera and make it easier to add new features)
#we will do the final tweaking and re-clean the DF after adding all the features of question 3

# Drop columns which contain redundunt information
rera = rera.drop(['country_full_x', 'country_full_y', 'country_abrv_x',
                  'confederation_x', 'confederation_y', 'country_abrv_y'], axis=1)
# drop rows which still contain missing data
rera.dropna(inplace=True)




#question 3
#3.1 adding average number of goals made in the last 5 matches
#I added two columns home_averageScore and away_averageScore

copy=rera

copy['Date'] = pd.to_datetime(rera['date'])


copy.sort_values(by=['home_team', 'date'], inplace=True)

# window =5 because we calculate the average during the last 5 matches
copy['home_averageScore'] = copy.groupby('home_team')['home_score'].rolling(window=5, min_periods=1).mean().reset_index(level=0, drop=True)
copy.sort_values(by=['away_team', 'date'], inplace=True)
copy['away_averageScore'] = copy.groupby('away_team')['away_score'].rolling(window=5, min_periods=1).mean().reset_index(level=0, drop=True)

#to_CSV(copy,"rera improved",False)














