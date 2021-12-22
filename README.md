# vizl

A tool to visualise live and historical data from the visl.org website


# ABOUT

This app is a result of a fun hobby project done to visualize, and in some cases analyse, publicly 
available data from the Vancouver Island Soccer League website [visl.org](https://visl.org/). The project was born out of 
a personal desire to build machine learning based fixtures prediction models until the volume of data available, its quality and the nature of the game itself made it impossible to build a reliable model. A basic ML model with 80% accuracy was built for just one team, but it has been accurate only 20% of the time in the real world so far! I have a long list of things to do in my head to imporve the app. Some of them are listed below. If you would like to see more features or contribute in any way, you are more than welcome!

# USE CASES 

-   For the visl fanatics, you can use this app to get the latest and historical fixtures, results, and standings
-   For coaches, you can use the app to get a quick picture of your or team's opponents performance in the current or previous seasons
-   For investors, this app can help you buy top players, sponsor teams in the league
-   Check out the latest news and twitter feeds from the league
-   Check fixture predictions at least for the top division 
-   Know which teams park the bus
-   Check if Ronaldo plays part time in the league
-   Check if Messi plays part time in the league. Currently impossible. Match MVPs used to be a thing on the visl website, but it is not in recent years
-   Check if a Sergio Ramos is in the league. I suspect there could be a few :)
-   Check if Manuel Neuer or Navas or one the the ellite goalies is here  in the league
-   Know patterns of players migration within the league
-   Know field locations of the games, which field is home to a team etc
-   Study inter-season changes and trends such as number of teams across years, number of goals, goals per game played, total goals scored or conceeded by teams as a fraction of totals games played etc

# TODO LIST

-   More work needed to handle edge cases, specially with the pool selection feature.
-   More work to do to increase efficiency in data handling and processing. The framework, streamlit, used is the fastest way to create data apps. It also comes with a small caveat for some as it re-runs all or most ( not sure here) of the codes when a parameter is changed. That slows down things a bit, but definately not that slow. Plus, in addtion to the default caching when applicable almost all functions used are made to cache thier output. 
-   Fully automated testing
-   Add a sidebar to select a team
-   Add a sidebar to select a player
-   Add a sidebar to select a fixture
-   Add prediction for the game. This has been almost impossible to do. Will try again for the top division where data are cleaner and more complete.
-   Allow users to edit table data or enter their own to experiment with the plots
-   Allow users multiple selections. For example two teams or two divisions
-   The chord diagrams are both beautiful and ugly at the same time. Fix what you can fix.
-   Make the app multipage 
-   Use teamcrests
-   Add more tables such as the one for discipline
-   Finally, add beer analytics. Beer is one of the main drivers of the league :joy:  



# A Note on Data 

The visl league is over 100 years old (established 1895). Some of the teams today are almost as old as the league itself. Some are only a couple of years young.  The league is a smoothly run machine, well managed by great individuals. The following observations and any recommendations given are meant for further improvements with respect to data quality. 

# Observations and Recommendations

## Observations 

- Not much data publicly available, at least on the visl website
- Noticeable team attrition rates, even within the limited data checked
- Teams tend to change their base names too often, and sometimes for no apparent reason. One good example is Gorge Us Guys turned into Gorge Us-Guys. A very strong team. They would like to keep their history in the league more than others

## Recommendations

-  Make more data easily accessible to the public
-  Encourage teams to remain in the league by providing the support they need. Additional fundings from local businesses and governments may be required. Canada is on the verge of qualifying for the FIFA World Cup 2022. Without amatuer leagues like the visl, that wouldn't have been possible.
-  Encourage teams to keep their base names for historical record keeping reasons. It is understood sponsors have a strong say in team names in amateur leagues, but there must be an alternative to changing a team's name significantly. Sponsors are part of the league's history, and they might be willing to compromise if they are made aware of the issue. Another solution to this problem would be using special tags to teams. That way teams can change their public names to meet sponsor requirements while still being identified as the same team to the league by their tags.


