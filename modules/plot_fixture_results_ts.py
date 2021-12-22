import streamlit as st
import plotly.express as px
import pandas as pd

# @st.cache(allow_output_mutation=True)
def home_or_away_results(fixture_results, home_or_away, team_colors, hover_data,selected_team):
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
    fg.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')

    fg.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGray',
                    title=f'Goal Diff. (for selected team - against)')
    fg.update_traces(marker=dict(size=15)
                        )

    fg.update_layout(coloraxis_colorbar=dict(
        # title="Season",
        title=f'Both Home & Away Game Results for {selected_team}.',
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
                    x0=min(fixture_results.Date),
                    y0=0,
                    x1=max(fixture_results.Date),
                    y1=0,
                    line=dict(color='red', width=1))
    fg.update_yaxes(range=[fixture_results.min(), fixture_results.max()])

    return fg

if __name__ == "__main__":
    import get_visl_fixture_and_standing_tables as gvfst
    import copy
    dv = '1'
    # Extract standing and fixture table to display
    fixtures, _ = gvfst.get_visl_fixture_and_standing_tables(dv)

    fixtures_ = copy.deepcopy(fixtures)

    fixtures_['Goal Difference (Home Team - Visiting Team )'] = fixtures_['HomeTeamScore'] - fixtures_[
        'VisitingTeamScore']
    fixtures_['Goal Difference (Visiting Team - Home Team )'] = fixtures_.VisitingTeamScore - fixtures_.HomeTeamScore

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
    st.plotly_chart(both_fixture_results_fig, use_container_width=True)