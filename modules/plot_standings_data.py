import streamlit as st
import plotly.express as px
import pandas as pd
import numpy as np
import random


@st.cache
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

if __name__ == '__main__':
    
    import get_visl_fixture_and_standing_tables as gvfs
    dv = '1'
    # Extract standing and fixture table to display
    fixtures, standings = gvfs.get_visl_fixture_and_standing_tables(dv)
    # process the standing tables for better display
    standings_display = standings.reset_index().rename(columns={'index': 'Rank'})
    standings_display['Rank'] = standings_display['Rank'] + 1
    standings_display.Season = standings_display.Season.apply(lambda x: x.strftime('%Y')).apply(
        lambda x: x + f'/{int(x) + 1}')

    st.write(standings_display)
    st.write(fixtures)
    st.plotly_chart(standings_plot(standings_display))  