
import streamlit as st
import plotly.express as px

@st.cache_data
def plot_goal_funnel(df, dv, yr, team_color_map):
    '''
    
    '''
    df = df.rename(columns={f'Goals_div_{dv}': 'Goals', f'Player_Name_div_{dv}': f'Player Name',
                            f'Team_Name_div_{dv}': 'Team Name'})
    fig = px.funnel(df[df.index.year == yr],
                    x='Player Name',
                    y='Goals',
                    color='Team Name',
                    color_discrete_map=team_color_map,
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


if __name__ == '__main__':
    import get_visl_goals_data as gvgd
    import random

    dv = '1'
    df = gvgd.get_visl_goals_data(dv)

    clr_map_keys = set(df[f'Team_Name_div_{dv}'].values)
    number_of_colors = len(df[f'Team_Name_div_{dv}'].unique())

    clr_map_values = ["#" + ''.join([random.choice('0123456789ABCDEF') for j in range(6)])
                    for i in range(number_of_colors)]
    team_colors = {i: j for i, j in zip(clr_map_keys, clr_map_values)}
    yr = 2020
    fig = plot_goal_funnel(df, dv, yr, team_colors)
    st.plotly_chart(fig)