import streamlit as st
import pandas as pd


# team summary stats with multi-index
@st.cache_data
def shutout_stats(df, dv):
    '''
    Function to  analyse shutout statisitcs of each team
    INPUT: 
    df: data frame created from the  shutout  Leader board
    dv = int: division level
    pool str: A, B, ...
    Returns: team shutout stats with multi-index for the division and pool
    '''
    team_stats = pd.DataFrame()
    for yr in range(2018, 2022):
        for team_name in df[df.index.year == yr][f'Team_Name_div_{dv}']:
            try:
                to_display = df[df.index.year == yr].query(f"Team_Name_div_{dv} == '{team_name}'").rename(
                    columns={f'Shutouts_div_{dv}': f'{yr}/{yr + 1} {team_name}'}).describe().round(2)
            except KeyError:
                st.write(f'{team_name} has a special character that I cannot parse at this time. Skipping it.')
                pass

            team_stats = pd.concat([team_stats, to_display], axis=1)
    sns = [x.split(' ')[0] for x in team_stats.T.index]
    tnm = [' '.join(x.split(' ')[1:]) for x in team_stats.T.index]
    team_stats = team_stats.T
    team_stats['Season'] = sns
    team_stats['Team'] = tnm
    team_stats.set_index(['Season', 'Team'], inplace=True)
    return team_stats

if __name__ == '__main__':
    import get_visl_shutout_data as gvsd
    dv = '1'
    df = gvsd.get_visl_shutout_data(dv)
    team_stats = shutout_stats(df, dv)
    st.write(team_stats)