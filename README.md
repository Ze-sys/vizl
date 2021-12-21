# vizl

A tool to visualise live and historical data from the visl.org website


# ABOUT

This app is  a result of a hobby project done to visualize, and in some cases analyse, publicly 
available data from the Vancouver Island Soccer League website [visl.org](https://visl.org/). I have a long 
list of things to do in my head to imporve it. Some of them are listed below. If you would like to see more features 
or contribute in any way, the best place to reach me is on the project's repository at 
[https://github.com/Ze-sys/vizl](https://github.com/Ze-sys/vizl)


# USE CASES 

-   For visl fanatics, you can use the app to get the latest standings and fixtures and results
-   For coaches, you can use the app to get a quick picture of your or other team's performance over time
-   Buy top players from the top teams in the league
-   Sponsor your favorite team
-   Check out the latest news from the league
-   See fixtures predictions
-   Check if Ronaldo plays part time in the league
-   Check if Messi plays part time in the league. Currently impossible. Match MVPs used to be a thing on the visl website, but it is not in recent years
-   Know patterns of player migration within the league
-   Know field locations of the games
-   Know field allocations, which field is home to a team etc
-   Study inter-season changes and trends such as number of teams across years, number of goals, goals per game played, total goals for or against teams as a fraction of totals etc

# TODO LIST

-   Break down section into modules
-   DRY : Use more classes to avoid code duplication
-   Add a sidebar to select the division
-   More work needed to handle edge cases, specially with the pool selection feature
-   Add a sidebar to select the year
-   Add a sidebar to select the team
-   Add a sidebar to select the player
-   Add a sidebar to select the fixture
-   Add a sidebar to select the season
-   Add a sidebar to select the game
-   Add prediction for the game. This has been almost impossible to do with the amount and quality of data
-   Allow users to select the year and division
-   Add more data download links specially for the figures shown
-   Allow users to enter their own data
-   Search workflow table to confirm the DATA ticket is entered
-   Allow users multiple selections
-   The chord diagrams are both beautiful and ugly at the same time. Fix what you can fix.
-   multiple pages/instances of this app
-   Add analytics on Beer, one of the main drivers of the league :joy:  



# A Note on Data 

The visl league is over 100 years old (established 1895). Some of the teams today are almost as old as the league itself. Some are only a couple of years young.  The league is a smoothly run machine, well managed by great individuals. The following observations and any recommendations given are meant for further improvements with respect to data quality. 

# Observations and Recommendations

## Observations 

- Not much data publicly available, at least on the visl website
- Noticeable team attrition rates, even within the limited data checked
- Teams tend to change their base names too often, and sometimes for no apparent reason. One good example is Gorge Us Guys turned into Gorge Us-Guys. A very strong team. They would like to keep their history in the league more than others

## Recommendations

-  Make more data easily accessible to the public
-  Encourage teams to remain in the league by providing the support they need. Additional fundings from local businesses and governments may be required. Canada is on the verge of qualifying for the FIFA World Cup 2022. Without amatuer leagues like the visl, that wouldn't have been possible
-  Encourage teams to keep their base names for historical record keeping reasons. It is understood sponsors have a strong say in team names in amateur leagues, but there must be an alternative to changing a team's name significantly. Sponsors are part of the league's history, and they may be willing to compromise if they are made aware of the issue. Another solution to this problem would be using special tags to teams. That way teams can change their public names to meet sponsor requirements while still being identified as the same team to the league by their tags.


