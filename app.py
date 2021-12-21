
import streamlit as st
import re
import copy
import random
import datetime
import numpy as np
import pandas as pd
import holoviews as hv
import plotly.express as px
from holoviews import opts, dim
import plotly.graph_objects as go
import streamlit.components.v1 as components
# from bokeh.sampledata.les_mis import data
import matplotlib as mpl

cmap = mpl.cm.jet
cmap_r = cmap.reversed()
hv.extension('bokeh')
hv.output(size=1500)

st.set_page_config(layout="wide")
pd.set_option('display.max_colwidth', None)


# app header
def header(str):
    '''
    '''
    st.markdown(f'<h1 style="color: #42f57b;font-size:80px;border-radius:50%;">{str}</h1>',
                unsafe_allow_html=True)


header('vizl')
st.write(
    f'<h1 style="color: #754df3;font-size:15px;border-radius:0%;">A tool to visualize live and historical data from the visl.org website</h1>',
    unsafe_allow_html=True)


def remove_special_chars(x):
    return x.replace("'", '')


# Allow uses to download the data shown in the app


class CAPTION:
    '''
    Class to hold all the captions for the tables
    Sets caption handles
    '''

    def __init__(self, dict_):
        self.caption = dict_.get('caption')
        self.data = dict_.get('data')
        self.col_to_index = dict_.get('index')  # column name of the data frame to be shown as index
        self.col_to_drop = dict_.get('drop')  # column name of the data frame to be dropped
        self.caption_handle = st.columns(1)

        '''
        Add more captions here to be show for figures
        '''


def write_table(X, width=None, height=None):
    '''
    Function to write a table to the app
    INPUT:
    X: data frame
    width: width of the table
    height: height of the table
    '''
    st.dataframe(X.data.set_index(X.col_to_index, inplace=False).drop(columns=X.col_to_drop).style.format(precision=0),
                 width, height)


@st.cache
def convert_df(df):
    return df.to_csv().encode('utf-8')


def csv_downloader(X):
    st.markdown(
        f'''<h6 style="color:white;font-size:16px;border-radius:0%;background-color:#754DF3;"><br> {X.caption}</h6></br>''',
        unsafe_allow_html=True)
    st.download_button(
        "Download Data",
        convert_df(X.data),
        X.caption + '.csv',
        "text/csv",
        key='download-csv_format'
    )


# yrr = st.sidebar.selectbox(label='Select year', options=[2019,2020,2021,2022])
dv = st.sidebar.selectbox(label='select a division', options=['div 1', 'div 2', 'div 3', 'div 4', 'div m'])

div_dict = {'div 1': '1',
            'div 2': '2',
            'div 3': '3',
            'div 4': '4',
            'div m': 'm'}
if dv:
    dv = div_dict.get(dv)


# if yrr and dv and pool:

@st.cache
def get_visl_shutout_data(dv):
    '''
    
    '''
    all_tables = pd.DataFrame([])
    for reg_yrs in range(2019, 2023):

        url = f'https://visl.org/webapps/spappz_live/division_goalie_stats?reg_year={reg_yrs}&division={dv}&sched_type=&combined=&sortby='
        table = pd.read_html(url, match='Player Name')
        year_idx = pd.Timestamp(year=reg_yrs - 1, month=9, day=1)

        for i in range(1, len(table)):
            if table[i].shape[0] > 2:  # ignore tables wihtout entry
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


df = get_visl_shutout_data(dv)


@st.cache
def get_visl_goals_data(dv):
    '''
    
    '''

    all_tables = pd.DataFrame([])
    for reg_yrs in range(2019, 2023):

        url = f'https://visl.org/webapps/spappz_live/division_player_stats?reg_year={reg_yrs}&division={dv}&sched_type=&combined=&sortby='
        table = pd.read_html(url, match='Goals')
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


df_goal = get_visl_goals_data(dv)


def plot_shutout_bar(df, dv):
    '''
    Function to plot bar charts of total golas scored per season
    INPUT: 
    df: data frame created from the  shutout Scorers Leader board
    dv = int: division level
    pool str: A, B, ...
    Returns: bar chart of total Shutouts per season for the division and pool
    '''
    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Middle-aligned",
        y=df.groupby('Year').sum()[f'Shutouts_div_{dv}'], x=list(set(df.index.year)),
        width=.5
    )
    )

    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=20, r=0, b=20, t=20),
        font_size=9,
        title=f'Division {dv} Total Shutouts',
        height=250,
        width=350,
        yaxis=dict(title='Total Shutouts'),
        xaxis=dict(
            title='Season',
            tickmode='array',
            tickvals=[i for i in range(2018, 2022)],
            ticktext=[f'{i}/{i + 1}' for i in range(2018, 2022)]
        )
    )
    return fig


def plot_goals_bar(df, dv):
    '''
    Function to plot bar charts of total golas scored per season
    INPUT: 
    df: data frame created from the  shutout Scorers Leader board
    dv = int: division level
    pool str: A, B, ...
    Returns: bar chart of total Shutouts per season for the division and pool
    '''
    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Middle-aligned",
        y=df.groupby('Year').sum()[f'Goals_div_{dv}'], x=list(set(df.index.year)),
        width=.5
    )
    )

    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=20, r=0, b=20, t=20),
        font_size=9,
        title=f'Division {dv} Total Goals',
        height=250,
        width=350,
        yaxis=dict(title='Total Goals'),
        xaxis=dict(
            title='Season',
            tickmode='array',
            tickvals=[i for i in range(2018, 2022)],
            ticktext=[f'{i}/{i + 1}' for i in range(2018, 2022)]
        )
    )
    return fig


# extract data based on user selection of pool
def team_div_pool_name(dv, pool):
    if dv != 'm':
        team_div_pool_name = f'(D{dv}{pool})'
        return bool(re.search('(D1|D2|D3A|D3B|D4A|D4B)', team_div_pool_name))
    elif dv == 'm' and pool == 'A' or pool == 'B':
        team_div_pool_name = f'(M{pool})'
        return bool(re.search('(MA|MB)', team_div_pool_name))


@st.cache
def filter_pool(df, dv, pool):
    '''
    Function to filter the data frame based on the pool selected
    INPUT: 
    df: data frame created from the  shutout Scorers Leader board
    pool str: A, B, ...
    Returns: data frame filtered based on the pool selected
    '''
    df = df[df[f'Team_Name_div_{dv}'].apply(lambda x: team_div_pool_name(dv, pool))]
    df = df.reset_index().rename(columns={'Year': 'Season'})
    df.Season = df.Season.apply(lambda x: x.strftime('%Y')).apply(lambda x: x + f'/{int(x) + 1}')

    return df


pool = st.sidebar.selectbox(label='select pool', options=['A', 'B'], )
df_shutout_pool = filter_pool(df, dv, pool)
df_goal_pool = filter_pool(df_goal, dv, pool)

fig_shutout_bar = plot_shutout_bar(df, dv)


# team stats full with multi-index
def div_shutout_stats(df, dv):
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


team_stats = div_shutout_stats(df, dv)
# st.dataframe(team_stats)


clr_map_keys = set(df[f'Team_Name_div_{dv}'].values)
number_of_colors = len(df[f'Team_Name_div_{dv}'].unique())

clr_map_values = ["#" + ''.join([random.choice('0123456789ABCDEF') for j in range(6)])
                  for i in range(number_of_colors)]
team_colors = {i: j for i, j in zip(clr_map_keys, clr_map_values)}


def plot_shutout_funnel(df, dv, yr):
    '''
    
    '''
    df = df.rename(columns={f'Shutouts_div_{dv}': 'Shutouts', f'Player_Name_div_{dv}': f'Player Name',
                            f'Team_Name_div_{dv}': 'Team Name'})
    fig = px.funnel(df[df.index.year == yr],
                    x='Player Name',
                    y='Shutouts',
                    color='Team Name',
                    color_discrete_map=team_colors,
                    # template="simple_white"
                    )

    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        font_size=9,
        title=f'Division {dv} Goal Keepers with clean sheet(s): Season {yr}/{yr + 1}',
        height=250,
        width=1400,
        margin=dict(l=20, r=0, b=20, t=20),
        legend=dict(font_size=8,
                    title='',
                    orientation='h',
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1),
        showlegend=False  # no need to show it since it is there from the goal funnel plots
    )
    return fig


def plot_goal_funnel(df, dv, yr):
    '''
    
    '''
    df = df.rename(columns={f'Goals_div_{dv}': 'Goals', f'Player_Name_div_{dv}': f'Player Name',
                            f'Team_Name_div_{dv}': 'Team Name'})
    fig = px.funnel(df[df.index.year == yr],
                    x='Player Name',
                    y='Goals',
                    color='Team Name',
                    color_discrete_map=team_colors,
                    # template="simple_white"
                    )

    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        font_size=9,
        title=f'Division {dv} Goal Scorers: Season {yr}/{yr + 1}',
        height=450,
        width=1400,
        margin=dict(l=20, r=0, b=20, t=180, pad=0),
        legend=dict(font_size=8,
                    title='',
                    orientation='h',
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1)
    )
    return fig


# the 4th table from top is the table that has standings data
def get_visl_standing_table(website_tables):
    return website_tables[4].rename(
        columns={x: y for x, y in zip(range(8), ['Team', 'GP', 'W', 'D', 'L', 'GF', 'GA', 'PTS'])}).query(
        f"Team != 'Team'").reset_index(drop=True)


# the 2nd from last table has fixtures data
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


# Extract standing and fixture table to display
fixtures, standings = get_visl_fixture_and_standing_tables(dv)

standings_display = standings.reset_index().rename(columns={'index': 'Rank'})
standings_display['Rank'] = standings_display['Rank'] + 1
standings_display.Season = standings_display.Season.apply(lambda x: x.strftime('%Y')).apply(
    lambda x: x + f'/{int(x) + 1}')

# Display fixtures tables
caption = f'Division {dv} Latest Fixtures and Results'
display_data = fixtures[fixtures.Result != '-'].drop(columns=['HomeTeamScore', 'VisitingTeamScore']).sort_values(
    by=['Date', 'Result'], ascending=False)

# use class to display the data table, its caption and a download button
C = CAPTION({'caption': caption, 'data': display_data.head(10), 'index': display_data.head(10).Date,
             'drop': display_data.Date.name})
csv_downloader(C)  # <-- this is the download button
write_table(C, None, 400)  # <-- this is the table

scheduled_fixtures_xpdr = st.expander('Show Scheduled Fixtures (if available)')
with scheduled_fixtures_xpdr:
    caption = f'Division {dv} Scheduled Fixtures'
    display_data = fixtures[fixtures.Result == '-'].drop(columns=['HomeTeamScore', 'VisitingTeamScore']).sort_values(
        by=['Date', 'Result'], ascending=False).reset_index(drop=True)

    C = CAPTION({'caption': caption, 'data': display_data, 'index': display_data.Date, 'drop': display_data.Date.name})
    csv_downloader(C)
    write_table(C, None, 100)

# Display standings tables

S = sorted(standings_display.Season.unique(), reverse=True)
caption = f'Division {dv} Standings: Season {S[0]}'
display_data = standings_display.query(f"Season == '{S[0]}'").sort_values(by=[f'Rank'],
                                                                          ascending=True).reset_index(drop=True)

C = CAPTION({'caption': caption, 'data': display_data, 'index': display_data.Rank, 'drop': display_data.Rank.name})
csv_downloader(C)  # <-- this is the download button
write_table(C, 1200, 400)  # <-- this is the table

past_seasons_standings_xpdr = st.expander(f'Show Division {dv} Standings for previous seasons (if available)')

with past_seasons_standings_xpdr:
    if len(S) > 1:
        for s in S[1:]:
            caption = f'Division {dv} Standings ({s})'
            display_data = standings_display.query(f"Season == '{s}'").sort_values(by=[f'Rank'],
                                                                                   ascending=True).reset_index(
                drop=True)
            C = CAPTION(
                {'caption': caption, 'data': display_data, 'index': display_data.Rank, 'drop': display_data.Rank.name})
            csv_downloader(C)
            write_table(C, 1200, 100)
    else:
        st.write(f'End of previous seasons data available for Division {dv}')

# Top goal scorers  and keepers tables

S = sorted(df_goal_pool.Season.unique(), reverse=True)

caption = f'Division {dv} Top Gaol Scorers ({S[0]})'
display_data = df_goal_pool.query(f"Season == '{S[0]}'").sort_values(by=[f'Goals_div_{dv}'],
                                                                     ascending=False).reset_index(drop=True)
C = CAPTION({'caption': caption, 'data': display_data, 'index': display_data.Season, 'drop': display_data.Season.name})
csv_downloader(C)
write_table(C, 1200, 100)

past_season_goal_pool_xpdr = st.expander(f'Show Division {dv} Top Goal Scorers for previous seasons (if available)')

# if season_goal_pool_checkbox:
with past_season_goal_pool_xpdr:
    if len(S) > 1:
        for s in S[1:]:
            caption = f'Division {dv} Top Goal Scorers ({s})'
            display_data = df_goal_pool.query(f"Season == '{s}'").sort_values(by=[f'Goals_div_{dv}'],
                                                                              ascending=False).reset_index(drop=True)
            C = CAPTION({'caption': caption, 'data': display_data, 'index': display_data.Season,
                         'drop': display_data.Season.name})
            csv_downloader(C)
            write_table(C, 1200, 100)
    else:
        st.write(f'End of previous seasons data available for Division {dv}')

S = sorted(df_shutout_pool.Season.unique(), reverse=True)
caption = f'Division {dv} Top Goal Keepers ({S[0]})'
display_data = df_shutout_pool.query(f"Season == '{S[0]}'").sort_values(by=[f'Shutouts_div_{dv}'],
                                                                        ascending=False).reset_index(drop=True)
C = CAPTION({'caption': caption, 'data': display_data, 'index': display_data.Season,
             'drop': display_data.Season.name})
csv_downloader(C)
write_table(C, 1200, 100)

past_season_shutout_pool_xpdr = st.expander(f'Show Division {dv} Top Goal Keepers for previous seasons (if available)')

with past_season_shutout_pool_xpdr:
    if len(S) > 1:

        for s in S[1:]:
            caption = f'Division {dv} Top Goal Keepers ({s})'
            display_data = df_shutout_pool.query(f"Season == '{s}'").sort_values(by=[f'Shutouts_div_{dv}'],
                                                                                 ascending=False).reset_index(drop=True)
            C = CAPTION({'caption': caption, 'data': display_data, 'index': display_data.Season,
                         'drop': display_data.Season.name})  # usually shorter table  as games added over time.
            csv_downloader(C)  # <-- this is the download button
            write_table(C, 1200, 100)  # <-- this is the table
        else:
            st.write(f'End of previous seasons data available for Division {dv}')


def standings_plot(standings_display):
    """

    """
    fig = px.scatter(
        standings_display.astype({'PTS': 'float64', 'W': 'float64', 'D': 'float64', 'L': 'float64', 'GF': 'float64'}),
        x='PTS', y='Rank', color='Season', size='GF', hover_data=['W', 'D', 'L'],
        hover_name="Team", color_continuous_scale=random.sample(px.colors.sequential.Turbo, 4)
    )
    for fig_data in fig.data: fig_data.update(mode='markers + lines')
    fig['layout']['yaxis']['autorange'] = "reversed"

    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        font_size=12,
        height=400,
        width=700,
        legend=dict(font_size=12, title="Season")
    )
    return fig


fig_standing = standings_plot(standings_display)

# plots

st.markdown(
    f'''<h6 style="color:white;font-size:16px;border-radius:0%;background-color:#754df3;"><br>Data 
    visualizations</h6</br>''',
    unsafe_allow_html=True)

# scatter plots
scatter_plots_xpdr = st.expander(f'Scatter Plots', expanded=True)
with scatter_plots_xpdr:

    st.markdown(
        f'''<h6 style="color:white;font-size:16px;border-radius:0%;background-color:None;"><br>Team standings vs maximum 
        points earned</h6></br>''',
        unsafe_allow_html=True)
    st.plotly_chart(fig_standing, use_container_width=True)

# bar plots
bar_plots_xpdr = st.expander(f'Bar Plots', expanded=True)
with bar_plots_xpdr:
    goals_cols = st.columns((3, 1))
    shutout_cols = st.columns((3, 1))
    yrr = st.sidebar.slider(label='select year', min_value=2018, max_value=2022, value=2019, )
    fig_shutout_funnel = plot_shutout_funnel(df, dv, yrr)
    fig_goal_funnel = plot_goal_funnel(df_goal, dv, yrr)

    fig_goal_bar = plot_goals_bar(df_goal, dv)
    if fig_goal_bar:
        goals_cols[1].plotly_chart(fig_goal_bar, use_container_width=True)

    if fig_shutout_bar:
        goals_cols[1].plotly_chart(fig_shutout_bar, use_container_width=True)

    if fig_goal_funnel:
        goals_cols[0].plotly_chart(fig_goal_funnel, use_container_width=True)
        # shutout_cols[0].plotly_chart(fig_shutout_funnel,use_container_width=True) # Not much info to show with a plot here

# Time series


fixtures_ = copy.deepcopy(fixtures)
fixtures_['Goal Difference (Home Team - Visiting Team )'] = fixtures_['HomeTeamScore'] - fixtures_[
    'VisitingTeamScore']
fixtures_['Goal Difference (Visiting Team - Home Team )'] = fixtures_.VisitingTeamScore - fixtures_.HomeTeamScore


def home_or_away_results(fixture_results, home_or_away, team_colors, hover_data, selected_team):
    if home_or_away == 'Away':
        fixture_results['Goal Difference (Home Team - Visiting Team )'] = fixture_results[
                                                                                'Goal Difference (Home Team - Visiting Team )'] * -1
    if home_or_away == 'Both':
        showlegend_ = True
        legend_title = 'Home Teams: '
    else:
        legend_title = ''
        showlegend_ = False

    fg = px.scatter(fixture_results, x='Date', y='Goal Difference (Home Team - Visiting Team )', color=team_colors,
                    hover_data=hover_data,
                    color_discrete_sequence=px.colors.qualitative.Alphabet,
                    )
    # fg['layout']['yaxis']['autorange'] = "reversed"
    fg.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')

    fg.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGray',
                    title=f'Goal Diff. (for selected team - against)')
    fg.update_traces(marker=dict(size=15)
                        )

    fg.update_layout(coloraxis_colorbar=dict(
        title="Season",
        thicknessmode="pixels", thickness=50,
        lenmode="pixels", len=300,
        yanchor="top", y=1,
        ticks="outside", ticksuffix="",
        dtick=10,
        tickvals=[i for i in range(2018, 2022)],
        ticktext=[f'{i}/{i + 1}' for i in range(2018, 2022)]
    ),
        width=1800, height=500,
        margin=dict(l=20, r=20, b=0, t=40),
        legend=dict(font_size=9,
                    title=legend_title,
                    orientation='h',
                    yanchor="bottom",
                    y=-.45,
                    xanchor="right",
                    x=1),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        showlegend=showlegend_,
        font_size=12,
        title=f'{home_or_away} Game Results',
    )

    fg.add_shape(type='line',
                    x0=min(fixtures_.Date),
                    y0=0,
                    x1=max(fixtures_.Date),
                    y1=0,
                    line=dict(color='red', width=1))
    fg.update_yaxes(range=[fixture_results.min(), fixture_results.max()])

    return fg


# PIE CHARTS
time_series_xpdr = st.expander(label='Time Series Plots', expanded=True)

# fixture_results_cols = st.columns([1, 1, 1, 3])
# time_series_cols = st.columns([5, 1])

with time_series_xpdr:
    both_team_selected = st.selectbox(label='Select a different Team',
                                                            options=sorted(list(set(fixtures_['HomeTeam']))))
    st.markdown(
        f'''<h1 style="color: green;font-size:12px;border-radius:100%;background-color:#3D0669;">Hover over a 
        marker to see game details. Click on the legend marker for the selected ({both_team_selected}) team to 
        hide its home game results. 
        Home Team markers above the zero line show a win for the selected ({both_team_selected}) team. Other Markers 
        above the zero line show the selected ({both_team_selected}) team lost the game.</h1>''',
        unsafe_allow_html=True)
    both_fixture_results_test = pd.concat([fixtures_.loc[fixtures_.VisitingTeam == both_team_selected].copy(),
                                            fixtures_.loc[fixtures_.HomeTeam == both_team_selected].copy()])
    both_fixture_results_fig = home_or_away_results(both_fixture_results_test, 'Both', 'HomeTeam',
                                                    ['VisitingTeam', 'HomeTeam', 'HomeTeamScore',
                                                        'VisitingTeamScore'], both_team_selected)
    st.plotly_chart(both_fixture_results_fig.update_layout(
        title=f'''Both Home & Away Game Results for {both_team_selected}.'''), use_container_width=True)


# PIE CHARTS
pie_xpdr = st.expander(label='Pie Charts', expanded=False)

with pie_xpdr:
    pie_cols = st.columns(2)
    fig_multi_pie = px.sunburst(standings_display, path=['Team', 'Season'], values='GF')
    fig_multi_pie.update_layout(margin=dict(t=30, l=0, r=0, b=0), title_text='Goals For', width=600, height=600)
    pie_cols[0].plotly_chart(fig_multi_pie)

    fig_multi_pie = px.sunburst(standings_display, path=['Team', 'Season'], values='GA')
    fig_multi_pie.update_layout(margin=dict(t=30, l=0, r=0, b=0), title_text='Goals Against', width=600, height=600)
    pie_cols[1].plotly_chart(fig_multi_pie)

# CHORDS

chord_data = fixtures[fixtures.Date.dt.year == yrr]

chord1 = hv.Chord(chord_data[['HomeTeam', 'Field']])
chord1.opts(fontscale=.75, width=45, height=45, title=f'Division {dv} Home Games Locations, season {yrr}/{yrr + 1}',
            label_text_font_size='8pt')
chord1.opts(
    opts.Chord(cmap='Category20', edge_cmap='Category20', edge_color=dim('Field').str(),
                labels='index', node_color=dim('index').str())
)

chord2 = hv.Chord(chord_data[['VisitingTeam', 'Field']])
chord2.opts(fontscale=.75, width=45, height=45, title=f'Division {dv} Away Games Locations, season {yrr}/{yrr + 1}',
            label_text_font_size='8pt')
chord2.opts(
    opts.Chord(cmap='Category20', edge_cmap='Category20', edge_color=dim('Field').str(),
                labels='index', node_color=dim('index').str())
)

chord1.opts(bgcolor='rgba(0,0,0,0)', padding=0.1,
            hooks=[lambda p, _: p.state.update(border_fill_color='rgba(0,0,0,0)')])
chord2.opts(bgcolor='rgba(0,0,0,0)', padding=0.1,
            hooks=[lambda p, _: p.state.update(border_fill_color='rgba(0,0,0,0)')])

hv.save(chord1 + chord2, 'tmp_html_files/chord_all_fixtures_results.html')
HtmlFile = open("tmp_html_files/chord_all_fixtures_results.html", 'r', encoding='utf-8')
source_code = HtmlFile.read()
html_background = "<style>:root {background-color:#754DF3;}</style>"
source_code = html_background + source_code
# st.write(source_code)
chord_xpdr = st.expander(label='Chord Diagrams', expanded=False)
with chord_xpdr:
    components.html(source_code, width=1400, height=730, scrolling=True)
    import folium
    from streamlit_folium import folium_static

    m = folium.Map(location=[48.4456201780829, -123.3645977], zoom_start=13, tiles='OpenStreetMap')

    folium.Marker(
        location=[48.4456201780829, -123.3645977],  # <- of Finny turf
        popup='Finlayson Turf',
        # More work to be done here.
        icon=folium.Icon(color="black", icon="glyphicon glyphicon-tree-conifer")
    ).add_to(m)

    folium_static(m, width=1400, height=360)


# USE CASES LIST

use_cases_list = [
    "For visl fanatics, you can use the app to get the latest standings and fixtures and results.",
    "For coaches, you can use the app to get a quick picture of your or other team's performance over time.",
    "Buy top players from the top teams in the league.",
    "Sponsor your favorite team.",
    "Check out the latest news from the league.",
    "See fixtures predictions.",
    "Check if Ronaldo plays part time in the league.",
    "Check if Messi plays part time the league. Currently impossible. Match MVPs used to be a thing on the visl website, but it is not in recent years.",
    "Know patterns of player migration within the league.",
    "Know field locations of the games.",
    "Know field allocations, which field is home to a team etc",
    "Study inter-season changes and trends such as number of teams across years, number of goals, goals per game played, total goals for or against teams as a fraction of totals etc"
]
xpdr = st.expander('Use cases', expanded=False)
with xpdr:
    for use_case in use_cases_list:
        st.markdown(f'<li style="color: green;font-size:12px;border-radius:50%;">- {use_case}</li>',
                    unsafe_allow_html=True)

# Improvements that may help keeping the league's history better

league_data_imporvements = st.expander("Observations and Recommendations")
league_data_observation_list = [
    "Not much data publicly available, at least on the visl website",
    "Noticeable team attrition rates, even within the limitted data checked.",
    "Teams tend to change thier base names too often, and sometimes for no apparent reason. One good example is Gorge Us Guys turned into Gorge Us-Guys. A very strong team. They would like to keep thier histor in the league more than others.",
]
league_data_imporvements_list = [
    "Make more data easily accessible to the public.",
    "Encourage teams to remain in the league by providing the support they need. Additional fundings from local businesses and governmants maybe required. Canada is on the verge of qualifying for the FIFA World Cup 2022. Without amatuer leagues like the visl, that wouldn't have been possible.",
    "Encourage teams to keep their base names for historical record keeping reasons. It is understood sponsors have a strong say in team names in amateur leageues, but there must be an alternative to changing a team's name significantly. Spornsors are part of the league's history, and they maybe willing to compormise if they are made awre of the issue. Another solution to this problem would be using special tags to teams. That way teams can change their public names to meet sponsor requirements while still being identified as the same team to the league by thir tags."
]
with league_data_imporvements:
    league_histry = """
    The visl league is over 100 years old (established 1895). Some of the teams today are almost as old as the league itself. Some are only a couple of years young. 
    The league is a smoothly run machine, well managed by great individuals. 
    The following observations and any recommendations given are meant for further improvements with respect to data quality. Othewise, the league is already awesome.  
    """
    st.markdown(f'<h8 style="color: #a366ff;font-size:16px;border-radius:0%;">{league_histry}</h8>',
                unsafe_allow_html=True)

    st.markdown(f'<h4 style="color: #42f57b;font-size:14px;border-radius:50%;"><br>Observations:</br></h4>',
                unsafe_allow_html=True)

    for obs in league_data_observation_list:
        st.markdown(f'<h8 style="color: white;font-size:8px;border-radius:0%;">- {obs}</h8>',
                    unsafe_allow_html=True)

    st.markdown(f'<h4 style="color: #42f57b;font-size:14px;border-radius:50%;"><br>Recommendations:</br></h4>',
                unsafe_allow_html=True)

    for impvt in league_data_imporvements_list:
        st.markdown(f'<h8 style="color: green;font-size:8px;border-radius:0%;">- {impvt}</h8>',
                    unsafe_allow_html=True)

# TODO LIST
todolist = [
    "Break down section into modules.",
    "DRY : Use more classess to avoide code duplication.",
    "Add a sidebar to select the division",
    "More work needed to handle edge cases, specially with the pool selection feature"
    "Add a sidebar to select the year",
    "Add a sidebar to select the team",
    "Add a sidebar to select the player",
    "Add a sidebar to select the fixture",
    "Add a sidebar to select the season",
    "Add a sidebar to select the game",
    "Add prediction for the game. This has been almost impossible to do with the amount and quality of data.",
    "Allow users to select the year and division",
    "Add more data download links specially for the figures shown",
    "Allow users to enter thier own data",
    "Search workflow table to confirm the DATA ticket is entered",
    "Allow users multiple selections",
    "The chord diagrams are both beautiful and ugly at the same time. Fix what you can fix."
    "multiple pages/instances of this app",
    "Add analytics on Beer, one of the main drivers of the league :)",
]
xpdr = st.expander('A To Do List')
with xpdr:
    for tdl in todolist:
        st.markdown(f'<li style="color: green;font-size:12px;border-radius:50%;">- [ ] {tdl}</li>',
                    unsafe_allow_html=True)

# ABOUT
about_xpdr = st.expander('About')

about_this_app = """This app is  a result of a hobby project done to visualize, and in some cases analyse, publicly 
available data from the Vancouver Island Soccer League website [https://visl.org/] (https://visl.org/). I have a long 
list of things to do in my head to imporve it. Some of them are listed below. If you would like to see more features 
or contribute in any way, the best place to reach the developer is on the project's repository at [
https://github.com/Ze-sys/predict_visl_winners](https://github.com/Ze-sys/vizl). """
with about_xpdr:
    st.markdown(f'<h8 style="color: green;font-size:16px;border-radius:100%;">{about_this_app}</h8>',
                unsafe_allow_html=True)
