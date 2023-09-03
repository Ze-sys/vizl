import streamlit as st
import pandas as pd
import numpy as np

@st.cache_data
def remove_special_chars(x):
    return x.replace("'", '')

@st.cache_data
def get_visl_goals_data(dv):
    '''
    
    '''

    all_tables = pd.DataFrame([])
    for reg_yrs in range(2019, 2023):

        url = f'https://visl.org/webapps/spappz_live/division_player_stats?reg_year={reg_yrs}&division={dv}&sched_type=&combined=&sortby='
        try:
            table = pd.read_html(url, match='Goals')
        except:
            st.write(f'No goal scorers data for division {dv} for  {reg_yrs} season') 
        
        year_idx = pd.Timestamp(year=reg_yrs - 1, month=9, day=1)

        for i in range(1, len(table)):
            if table[i].shape[0] > 1:  # ignore tables wihtout entry
                # A work around to skip rows showing top scoring player(s) photos:
                # take the ith table (table[i]), which usually contains all goal scorers, 
                # remove rows that contain the string "Current Goal Scoring" in the first col ([0]),
                # remove duplicates
                # reset index with drop true
                tbl = table[i][table[i][0].str.contains("Current Goal Scoring") == False].drop_duplicates().reset_index(
                    drop=True)
                df = pd.DataFrame(tbl)
                df = df.rename(columns=df.iloc[0]).drop(df.index[0])
                yr = np.tile(year_idx, (len(df), 1))
                df['Year'] = yr
                #             display(df)  # this would print each year and pool table data
                all_tables = pd.concat([all_tables, df], axis=0, ignore_index=True)
    all_tables[f'Team Name'] = all_tables[f'Team Name'].apply(lambda x: remove_special_chars(x))
    all_tables.rename(columns={'Player Name': f'Player_Name_div_{dv}', 'Team Name': f'Team_Name_div_{dv}',
                               'Goals': f'Goals_div_{dv}'}, inplace=True)
    all_tables.index = all_tables.Year
    all_tables.drop(columns='Year', inplace=True)
    all_tables[f'Goals_div_{dv}'] = all_tables[f'Goals_div_{dv}'].astype(int)

    return all_tables

if __name__ == '__main__':
    dv = '1'
    st.write(get_visl_goals_data(dv))