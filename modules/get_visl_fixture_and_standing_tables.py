
import streamlit as st
import re
import numpy as np
import pandas as pd


# the 4th table from top is the table that has standings data
@st.cache
def get_visl_standing_table(website_tables):
    return website_tables[4].rename(
        columns={x: y for x, y in zip(range(8), ['Team', 'GP', 'W', 'D', 'L', 'GF', 'GA', 'PTS'])}).query(
        f"Team != 'Team'").reset_index(drop=True)


# the 2nd from last table has fixtures data
@st.cache(allow_output_mutation=True) 
def get_visl_fixtures_table(website_tables):
    return website_tables[-2].dropna(axis=0, how='all')

@st.cache
def get_visl_fixture_and_standing_tables(dv):
    standings = pd.DataFrame(columns=['Season', 'Team', 'GP', 'W', 'D', 'L', 'GF', 'GA', 'PTS'])
    fixtures = pd.DataFrame(
        columns=['Schedule', 'Type', 'Date', 'Division', 'HomeTeam', 'Result', 'VisitingTeam', 'Field'])
    for reg_yrs in range(2019, 2023):
        website_tables = pd.read_html(
            f'https://visl.org/webapps/spappz_live/div_stats?reg_year={reg_yrs}&division={dv}&sched_pool=&sched_type=reg')
        standings_table = get_visl_standing_table(website_tables)
        standings_table['Season'] = pd.Timestamp(year=reg_yrs - 1, month=9, day=1)
        standings = pd.concat([standings, standings_table], axis=0)

        fixtures_table = get_visl_fixtures_table(website_tables)
        # only 7 cols are there
        fixtures_table = fixtures_table.iloc[:, 0:8]
        fixtures_table_header = ['Schedule', 'Type', 'Date', 'Division', 'HomeTeam', 'Result', 'VisitingTeam', 'Field']
        fixtures_table.columns = fixtures_table_header
        # cleanup of duplicate rows with game date and headers
        fixtures_table = fixtures_table.query(f"Type == 'League'").reset_index(drop=True)
        fixtures = pd.concat([fixtures, fixtures_table], axis=0)

    # get scores as int and put them in cols
    TeamScore = [re.findall(r'\d+', n) for n in fixtures.Result]
    fixtures['HomeTeamScore'] = [int(hts[0]) if hts else np.nan for hts in TeamScore]
    fixtures['VisitingTeamScore'] = [int(hts[1]) if hts else np.nan for hts in TeamScore]
    # convert the Date col as time stamp for plotting purpose
    fixtures.Date = fixtures.Date.apply(lambda x: pd.Timestamp(x))
    return fixtures, standings

if __name__ == '__main__':
    dv = '1'
    # Extract standing and fixture table to display
    fixtures, standings = get_visl_fixture_and_standing_tables(dv)
    st.write(standings)
    st.write(fixtures)