import streamlit as st
import pandas as pd
import numpy as np

@st.cache_data
def remove_special_chars(x):
    return x.replace("'", '')

@st.cache_data
def get_visl_shutout_data(dv):
    '''
    Grab the data from the visl website using pandas read html method
    INPUT: dv: division (see above the dict of divisions). In the furture, year or year range can be added
    OUTPUT: data frame
    '''
    all_tables = pd.DataFrame([])
    for reg_yrs in range(2019, 2025):
        if reg_yrs < 2022:
            url = f'https://visl.org/webapps/spappz_live/division_goalie_stats?reg_year={reg_yrs}&division={dv}&sched_type=reg&sortby='
        else:
            url = f'https://visl.org/webapps/spappz_live/division_goalie_stats?reg_year={reg_yrs}&division={dv}&sched_pool=A&sched_type=reg&firsttime=0'

        try:
            table = pd.read_html(url, match='Player Name')
        except:
            st.write(f'No shutouts data for division {dv} for  {reg_yrs} season')            

        year_idx = pd.Timestamp(year=reg_yrs - 1, month=9, day=1)

        for i in range(1, len(table)):
            if table[i].shape[0] > 2:  # ignore tables without entry
                df = pd.DataFrame(table[i])
                df = df.rename(columns=df.iloc[0]).drop(df.index[0])
                yr = np.tile(year_idx, (len(df), 1))
                df['Year'] = yr
                #             display(df)  # this would print each year and pool table data
                all_tables = pd.concat([all_tables, df], axis=0, ignore_index=True)
    all_tables[f'Team Name'] = all_tables[f'Team Name'].apply(lambda x: remove_special_chars(x))
    all_tables.rename(columns={'Player Name': f'Player_Name_div_{dv}', 'Team Name': f'Team_Name_div_{dv}',
                               'Shutouts': f'Shutouts_div_{dv}'}, inplace=True)
    all_tables.index = all_tables.Year
    all_tables.drop(columns='Year', inplace=True)
    all_tables[f'Shutouts_div_{dv}'] = all_tables[f'Shutouts_div_{dv}'].astype(int)

    return all_tables

if __name__ == '__main__':
    dv = '1'
    st.write(get_visl_shutout_data(dv))