
import streamlit as st
import re
import csv
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

# import the helper methods
import modules.get_visl_shutout_data as gvsd
import modules.get_visl_goals_data as gvgd
import modules.plot_goals_bar as pg
import modules.plot_shutout_bar as ps
import modules.plot_shutout_funnel as psf
import modules.plot_goal_funnel as pgf
import modules.shutouts_stats as ss
import modules.plot_standings_data as psd
import modules.get_visl_fixture_and_standing_tables as gvfst
import modules.plot_fixture_results_ts as pfrts


cmap = mpl.cm.jet
cmap_r = cmap.reversed()
hv.extension('bokeh')
hv.output(size=1500)

st.set_page_config(layout="wide")
pd.set_option('display.max_colwidth', None)


# app name
def app_name(str):
    '''
    '''
    st.markdown(f'<h1 style="color: #42f57b;font-size:80px;border-radius:50%;">{str}</h1>',
                unsafe_allow_html=True)

app_name('vizl')

st.write(
    f'<h1 style="color: #754df3;font-size:15px;border-radius:0%;">A tool to visualize live and historical data from the visl.org website</h1>',
    unsafe_allow_html=True)

@st.cache
def remove_special_chars(x):
    return x.replace("'", '')



class CAPTION:
    '''
    Class to hold all the captions for the tables and figures
    '''

    def __init__(self, dict_):
        self.caption = dict_.get('caption')
        self.data = dict_.get('data')
        self.col_to_index = dict_.get('index')  # column name of the data frame to be shown as index
        self.col_to_drop = dict_.get('drop')  # column name of the data frame to be dropped
        self.caption_handle = st.columns(1)

        '''
        Add more captions here to be shown for figures
        '''


def write_table(x, width=None, height=None):
    '''
    Function to write a table to the app
    INPUT:
    X: data frame
    width: width of the table
    height: height of the table
    '''
    st.dataframe(x.data.set_index(x.col_to_index, inplace=False).drop(columns=x.col_to_drop).style.format(precision=0),
                 width, height)
    

@st.cache
def df_to_csv(df):
    return df.to_csv().encode('utf-8')


def csv_downloader(x):
    st.markdown(
        f'''<h6 style="color:white;font-size:16px;border-radius:0%;background-color:#754DF3;"><br> {x.caption}</h6></br>''',
        unsafe_allow_html=True)
    st.download_button(
        "Download Data",
        df_to_csv(x.data),
        x.caption + '.csv',
        "text/csv",
        key='download-csv_format'
    )

dv = st.sidebar.selectbox(label='select a division', options=['div 1', 'div 2', 'div 3', 'div 4', 'div m'])

div_dict = {'div 1': '1','div 2': '2','div 3': '3', 'div 4': '4', 'div m': 'm'}
if dv:
    dv = div_dict.get(dv)

# get shutouts data table from the visl website
df = gvsd.get_visl_shutout_data(dv)
# get goals data table
df_goal = gvgd.get_visl_goals_data(dv)

# extract data based on user selection of pool
@st.cache
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
# bar plot for shutouts
fig_shutout_bar = ps.plot_shutout_bar(df, dv)
# team summary stats with multi-index
team_stats = ss.shutout_stats(df, dv)
# st.dataframe(team_stats)
# assign each team a unique color per session for some visualizations
clr_map_keys = set(df[f'Team_Name_div_{dv}'].values)
number_of_colors = len(df[f'Team_Name_div_{dv}'].unique())
clr_map_values = ["#" + ''.join([random.choice('0123456789ABCDEF') for j in range(6)])
                  for i in range(number_of_colors)]
team_colors = {i: j for i, j in zip(clr_map_keys, clr_map_values)}

# Extract standing and fixture table to display
fixtures, standings = gvfst.get_visl_fixture_and_standing_tables(dv)
# process the standing tables for better display
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
    write_table(C, None, 110)

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
            write_table(C, 1200, 110)
    else:
        st.write(f'End of previous seasons data available for Division {dv}')

# Top goal scorers  and keepers tables

S = sorted(df_goal_pool.Season.unique(), reverse=True)

caption = f'Division {dv} Top Goal Scorers ({S[0]})'
display_data = df_goal_pool.query(f"Season == '{S[0]}'").sort_values(by=[f'Goals_div_{dv}'],
                                                                     ascending=False).reset_index(drop=True)
C = CAPTION({'caption': caption, 'data': display_data, 'index': display_data.Season, 'drop': display_data.Season.name})
csv_downloader(C)
write_table(C, 1200, 110)

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
            write_table(C, 1200, 110)
    else:
        st.write(f'End of previous seasons data available for Division {dv}')

S = sorted(df_shutout_pool.Season.unique(), reverse=True)
caption = f'Division {dv} Top Goal Keepers ({S[0]})'
display_data = df_shutout_pool.query(f"Season == '{S[0]}'").sort_values(by=[f'Shutouts_div_{dv}'],
                                                                        ascending=False).reset_index(drop=True)
C = CAPTION({'caption': caption, 'data': display_data, 'index': display_data.Season,
             'drop': display_data.Season.name})
csv_downloader(C)
write_table(C, 1200, 110)

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
            write_table(C, 1200, 110)  # <-- this is the table
        else:
            st.write(f'End of previous seasons data available for Division {dv}')

# show plot of rankings for all seasons
fig_standing = psd.standings_plot(standings_display)

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
    fig_shutout_funnel = psf.plot_shutout_funnel(df, dv, yrr, team_colors)
    fig_goal_funnel = pgf.plot_goal_funnel(df_goal, dv, yrr,team_colors)

    fig_goal_bar = pg.plot_goals_bar(df_goal, dv)
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



# time series plots
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

    # both_fixture_results_fig =  pfrts.home_or_away_results                                   
    # both_fixture_results_fig = home_or_away_results(both_fixture_results_test, 'Both', 'HomeTeam',
    #                                                 ['VisitingTeam', 'HomeTeam', 'HomeTeamScore',
    #                                                     'VisitingTeamScore'], both_team_selected)

    both_fixture_results_fig = pfrts.home_or_away_results(both_fixture_results_test, 'Both', 'HomeTeam',
                                                    ['VisitingTeam', 'HomeTeam', 'HomeTeamScore',
                                                        'VisitingTeamScore'], both_team_selected)
    
    st.plotly_chart(both_fixture_results_fig.update_layout(
        title=f'''Both Home & Away Game Results for {both_team_selected}.'''), use_container_width=True)


# PIE CHARTS
pie_xpdr = st.expander(label='Pie Charts', expanded=False)

per_game_stats = copy.deepcopy(standings_display) 
per_game_stats['GFPG']= standings_display.GF.astype(int)/standings_display.GP.astype(int)
per_game_stats['GAPG']= standings_display.GA.astype(int)/standings_display.GP.astype(int)
per_game_stats['WPG']= standings_display.W.astype(int)/standings_display.GP.astype(int)
per_game_stats['LPG']= standings_display.L.astype(int)/standings_display.GP.astype(int)

with pie_xpdr:
    pie_cols = st.columns(2)
    fig_multi_pie = px.sunburst(standings_display, path=['Team', 'Season'], values='GF')
    fig_multi_pie.update_layout(margin=dict(t=30, l=0, r=0, b=0), title_text='Goals Scored', width=500, height=500)
    pie_cols[0].plotly_chart(fig_multi_pie)

    fig_multi_pie = px.sunburst(per_game_stats, path=['Team', 'Season'], values='GFPG')
    fig_multi_pie.update_layout(margin=dict(t=30, l=0, r=0, b=0), title_text='Goals Scored (Per Game Played)', width=500, height=500)
    pie_cols[0].plotly_chart(fig_multi_pie)

    fig_multi_pie = px.sunburst(per_game_stats, path=['Team', 'Season'], values='WPG')
    fig_multi_pie.update_layout(margin=dict(t=30, l=0, r=0, b=0), title_text='Wins (Per Game Played)', width=500, height=500)
    pie_cols[0].plotly_chart(fig_multi_pie)

    
    fig_multi_pie = px.sunburst(standings_display, path=['Team', 'Season'], values='GA')
    fig_multi_pie.update_layout(margin=dict(t=30, l=0, r=0, b=0), title_text='Goals Conceded', width=500, height=500)
    pie_cols[1].plotly_chart(fig_multi_pie)
    
    fig_multi_pie = px.sunburst(per_game_stats, path=['Team', 'Season'], values='GAPG')
    fig_multi_pie.update_layout(margin=dict(t=30, l=0, r=0, b=0), title_text='Goals Conceded(Per Game Played)', width=500, height=500)
    pie_cols[1].plotly_chart(fig_multi_pie)

    fig_multi_pie = px.sunburst(per_game_stats, path=['Team', 'Season'], values='LPG')
    fig_multi_pie.update_layout(margin=dict(t=30, l=0, r=0, b=0), title_text='Losses (Per Game Played)', width=500, height=500)
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
        icon=folium.Icon(color="darkpurple", icon="glyphicon glyphicon-tree-conifer")
    ).add_to(m)

    folium_static(m, width=1400, height=360)


# USE CASES LIST
uc = csv.reader(open('csv/use_cases.csv', 'r'), delimiter=';')
use_cases_list = list(uc)

xpdr = st.expander('Use cases', expanded=False)
with xpdr:
    for use_case in use_cases_list:
        st.markdown(f'<h6 style="color: white;font-size:12px;border-radius:50%;"> - {use_case[0]}</h6>',
                    unsafe_allow_html=True)


# ABOUT
about_xpdr = st.expander('About')

about_this_app = """This app is  a result of a hobby project done to visualize, and in some cases analyse, publicly 
available data from the Vancouver Island Soccer League website (https://visl.org/). I have a long 
list of things to do in my head to imporve the app. If you find it useful and would like to see more features 
or contribute in any way, the best place to reach me is on the project's repository at 
https://github.com/Ze-sys/vizl. """
with about_xpdr:
    st.markdown(f'<li style="color: #42f57b;font-size:12px;border-radius:50%;">{about_this_app}</li>',
                unsafe_allow_html=True)
