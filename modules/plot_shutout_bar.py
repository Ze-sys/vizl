import streamlit as st
import plotly.graph_objects as go

@st.cache_data
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

if __name__ == '__main__':

    import modules.get_visl_shutout_data as gvsd
    dv = '1'
    df = gvsd.get_visl_shutout_data(dv)
    fig = plot_shutout_bar(df, dv)
    st.plotly_chart(fig)