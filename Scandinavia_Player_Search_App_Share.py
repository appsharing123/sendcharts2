import matplotlib.pyplot as plt
import pandas as pd

import os
from natsort import natsorted
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from streamlit.hashing import _CodeHasher
import matplotlib.font_manager as fm
import numpy as np
import matplotlib.cm as cm
import matplotlib.colors as colors
from matplotlib.colors import TwoSlopeNorm
import matplotlib.cbook as cbook
import matplotlib.image as image
import matplotlib.gridspec as gridspec
import matplotlib as mpl
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.projections import get_projection_class





try:
    # Before Streamlit 0.65
    from streamlit.ReportThread import get_report_ctx
    from streamlit.server.Server import Server
except ModuleNotFoundError:
    # After Streamlit 0.65
    from streamlit.report_thread import get_report_ctx
    from streamlit.server.server import Server


@st.cache(allow_output_mutation=True)
def load_data():
    #url = 'https://drive.google.com/file/d/1KD5nxMlZZImiArxLXg4N43uiUkJ4taQP/view?usp=sharing'
    #path = 'https://drive.google.com/uc?export=download&id='+url.split('/')[-2]
    path = './Scandinavia.csv'
    df = pd.read_csv(path)
    #df = df[(df.Team.isin(player)) & (df.Season.isin(season))]
    return df

def load_CBmatchdata():
    #url = 'https://drive.google.com/file/d/1KD5nxMlZZImiArxLXg4N43uiUkJ4taQP/view?usp=sharing'
    #path = 'https://drive.google.com/uc?export=download&id='+url.split('/')[-2]
    path = './Scandinavia CB Match.csv'
    data = pd.read_csv(path)
    #df = df[(df.Team.isin(player)) & (df.Season.isin(season))]
    return data

def load_FBmatchdata():
    #url = 'https://drive.google.com/file/d/1KD5nxMlZZImiArxLXg4N43uiUkJ4taQP/view?usp=sharing'
    #path = 'https://drive.google.com/uc?export=download&id='+url.split('/')[-2]
    path = './Scandinavia FB Match.csv'
    data = pd.read_csv(path)
    #df = df[(df.Team.isin(player)) & (df.Season.isin(season))]
    return data




def main():
    state = _get_state()
    pages = {
        "CB Percentile": DPercentile,
        "FB Percentile": FBPercentile,
    }

    # st.sidebar.title("Page Filters")
    page = st.sidebar.radio("Select Page", tuple(pages.keys()))

    # Display the selected page with the session state
    pages[page](state)

    # Mandatory to avoid rollbacks with widgets, must be called at the end of your app
    state.sync()


def display_state_values(state):
    st.write("Input state:", state.input)
    st.write("Slider state:", state.slider)
    # st.write("Radio state:", state.radio)
    st.write("Checkbox state:", state.checkbox)
    st.write("Selectbox state:", state.selectbox)
    st.write("Multiselect state:", state.multiselect)

    for i in range(3):
        st.write(f"Value {i}:", state[f"State value {i}"])

    if st.button("Clear state"):
        state.clear()


def multiselect(label, options, default, format_func=str):
    """multiselect extension that enables default to be a subset list of the list of objects
     - not a list of strings

     Assumes that options have unique format_func representations

     cf. https://github.com/streamlit/streamlit/issues/352
     """
    options_ = {format_func(option): option for option in options}
    default_ = [format_func(option) for option in default]
    selections = st.multiselect(
        label, options=list(options_.keys()), default=default_, format_func=format_func
    )
    return [options_[format_func(selection)] for selection in selections]


# selections = multiselect("Select", options=[Option1, Option2], default=[Option2])


class _SessionState:

    def __init__(self, session, hash_funcs):
        """Initialize SessionState instance."""
        self.__dict__["_state"] = {
            "data": {},
            "hash": None,
            "hasher": _CodeHasher(hash_funcs),
            "is_rerun": False,
            "session": session,
        }

    def __call__(self, **kwargs):
        """Initialize state data once."""
        for item, value in kwargs.items():
            if item not in self._state["data"]:
                self._state["data"][item] = value

    def __getitem__(self, item):
        """Return a saved state value, None if item is undefined."""
        return self._state["data"].get(item, None)

    def __getattr__(self, item):
        """Return a saved state value, None if item is undefined."""
        return self._state["data"].get(item, None)

    def __setitem__(self, item, value):
        """Set state value."""
        self._state["data"][item] = value

    def __setattr__(self, item, value):
        """Set state value."""
        self._state["data"][item] = value

    def clear(self):
        """Clear session state and request a rerun."""
        self._state["data"].clear()
        self._state["session"].request_rerun()

    def sync(self):
        """Rerun the app with all state values up to date from the beginning to fix rollbacks."""

        # Ensure to rerun only once to avoid infinite loops
        # caused by a constantly changing state value at each run.
        #
        # Example: state.value += 1
        if self._state["is_rerun"]:
            self._state["is_rerun"] = False

        elif self._state["hash"] is not None:
            if self._state["hash"] != self._state["hasher"].to_bytes(self._state["data"], None):
                self._state["is_rerun"] = True
                self._state["session"].request_rerun()

        self._state["hash"] = self._state["hasher"].to_bytes(self._state["data"], None)


def _get_session():
    session_id = get_report_ctx().session_id
    session_info = Server.get_current()._get_session_info(session_id)

    if session_info is None:
        raise RuntimeError("Couldn't get your Streamlit Session object.")

    return session_info.session


def _get_state(hash_funcs=None):
    session = _get_session()

    if not hasattr(session, "_custom_session_state"):
        session._custom_session_state = _SessionState(session, hash_funcs)

    return session._custom_session_state


def DPercentile(state):
    df = load_data()
    pdf = load_CBmatchdata()

    df['SuccessfulDActions/90'] = df['Successful defensive actions per 90'].rank(pct=True)
    df['Defensive%'] = df['Defensive duels won, %'].rank(pct=True)
    df['AerialDuels/90'] = df['Aerial duels per 90'].rank(pct=True)
    df['Aerial%'] = df['Aerial duels won, %'].rank(pct=True)
    df['PAdjInterceptions'] = df['PAdj Interceptions'].rank(pct=True)
    df['Fouls/90'] = df['Fouls per 90'].rank(pct=True)
    df['Passes/90'] = df['Passes per 90'].rank(pct=True)
    df['Pass%'] = df['Accurate passes, %'].rank(pct=True)
    df['ForwardPasses/90'] = df['Forward passes per 90'].rank(pct=True)
    df['xA/90'] = df['xA per 90'].rank(pct=True)
    df['KeyPass/90'] = df['Key passes per 90'].rank(pct=True)
    df['PenAreaEntries/90'] = df['Passes to penalty area per 90'].rank(pct=True)
    df['PenAreaPass%'] = df['Accurate passes to penalty area, %'].rank(pct=True)
    df['DeepCompletions/90'] = df['Deep completions per 90'].rank(pct=True)
    df['ThroughBalls/90'] = df['Through passes per 90'].rank(pct=True)
    df['ProgressivePass/90'] = df['Progressive passes per 90'].rank(pct=True)
    df['ShotAssists/90'] = df['Shot assists per 90'].rank(pct=True)
    df['ShotsBlocked/90'] = df['Shots blocked per 90'].rank(pct=True)
    df['ForwardPass%'] = df['Accurate forward passes, %'].rank(pct=True)

    totaldf = pd.DataFrame(data=df, columns=['Team', 'Player', 'Age', 'Position', 'Passport country', 'Passes/90', 'Pass%',
                                        'ForwardPass%', 'ForwardPasses/90', 'ProgressivePass/90',
                                        'SuccessfulDActions/90',
                                        'Defensive%', 'AerialDuels/90', 'Aerial%', 'ShotsBlocked/90',
                                        'PAdjInterceptions', 'Fouls/90'])

    D = pd.DataFrame(data=df, columns=['Team', 'Player', 'Age', 'Position', 'Passport country', 'Passes/90', 'Pass%',
                                        'ForwardPass%', 'ForwardPasses/90', 'ProgressivePass/90',
                                        'SuccessfulDActions/90',
                                        'Defensive%', 'AerialDuels/90', 'Aerial%', 'ShotsBlocked/90',
                                        'PAdjInterceptions', 'Fouls/90'])


    player = st.sidebar.selectbox('Select Player', natsorted(df.Player.unique()))

    D = D[D['Player'] == (player)]
    test = D[D['Player'] == player]
    test = test.drop(columns=['Team', 'Age', 'Position', 'Passport country'])
    test = test.set_index('Player')
    test = test.transpose()
    test.head()

    ax = test.plot.barh(figsize=(24, 20))
    ax.set_xlim([0, 1])
    ax.set_xticks([0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1])
    ax.tick_params(axis='both', which='major', labelsize=18)
    ax.xaxis.grid(True, linestyle='--', which='major',
                  color='black', alpha=.5)
    ax.axvline(.5, color='red', alpha=0.5, lw=10)  # median position
    ax.set_facecolor('#E6E6E6')
    plt.title(str(player) + '\nCB Percentile Chart', weight='bold',size=24)
    st.pyplot(plt)

    playerdf = totaldf[totaldf['Player'] == (player)]
    st.dataframe(playerdf)

    st.text('')
    st.title('CBs with Avg Percentile Greater Than 0.75')
    st.table(pdf)


def FBPercentile(state):
    df = load_data()
    pdf = load_FBmatchdata()

    df['SuccessfulDActions/90'] = df['Successful defensive actions per 90'].rank(pct=True)
    df['Defensive%'] = df['Defensive duels won, %'].rank(pct=True)
    df['AerialDuels/90'] = df['Aerial duels per 90'].rank(pct=True)
    df['Aerial%'] = df['Aerial duels won, %'].rank(pct=True)
    df['PAdjInterceptions'] = df['PAdj Interceptions'].rank(pct=True)
    df['Fouls/90'] = df['Fouls per 90'].rank(pct=True)
    df['Passes/90'] = df['Passes per 90'].rank(pct=True)
    df['Pass%'] = df['Accurate passes, %'].rank(pct=True)
    df['ForwardPasses/90'] = df['Forward passes per 90'].rank(pct=True)
    df['xA/90'] = df['xA per 90'].rank(pct=True)
    df['KeyPass/90'] = df['Key passes per 90'].rank(pct=True)
    df['PenAreaEntries/90'] = df['Passes to penalty area per 90'].rank(pct=True)
    df['PenAreaPass%'] = df['Accurate passes to penalty area, %'].rank(pct=True)
    df['DeepCompletions/90'] = df['Deep completions per 90'].rank(pct=True)
    df['ThroughBalls/90'] = df['Through passes per 90'].rank(pct=True)
    df['ProgressivePass/90'] = df['Progressive passes per 90'].rank(pct=True)
    df['ShotAssists/90'] = df['Shot assists per 90'].rank(pct=True)
    df['ShotsBlocked/90'] = df['Shots blocked per 90'].rank(pct=True)
    df['ForwardPass%'] = df['Accurate forward passes, %'].rank(pct=True)

    totaldf = pd.DataFrame(data=df,
                           columns=['Team', 'Player', 'Age', 'Position', 'Passport country', 'Passes/90', 'Pass%',
                                    'ForwardPass%', 'ForwardPasses/90', 'ProgressivePass/90', 'KeyPass/90', 'xA/90',
                                    'PenAreaEntries/90', 'ShotAssists/90',
                                    'SuccessfulDActions/90', 'Defensive%', 'Aerial%', 'PAdjInterceptions','Fouls/90'])

    D = pd.DataFrame(data=df, columns=['Team', 'Player', 'Age', 'Position', 'Passport country', 'Passes/90', 'Pass%',
                                    'ForwardPass%', 'ForwardPasses/90', 'ProgressivePass/90', 'KeyPass/90', 'xA/90',
                                    'PenAreaEntries/90', 'ShotAssists/90',
                                    'SuccessfulDActions/90', 'Defensive%', 'Aerial%', 'PAdjInterceptions','Fouls/90'])

    player = st.sidebar.selectbox('Select Player', natsorted(df.Player.unique()))

    D = D[D['Player'] == (player)]
    test = D[D['Player'] == player]
    test = test.drop(columns=['Team', 'Age', 'Position', 'Passport country'])
    test = test.set_index('Player')
    test = test.transpose()
    test.head()

    ax = test.plot.barh(figsize=(24, 20))
    ax.set_xlim([0, 1])
    ax.set_xticks([0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1])
    ax.tick_params(axis='both', which='major', labelsize=18)
    ax.xaxis.grid(True, linestyle='--', which='major',
                  color='black', alpha=.5)
    ax.axvline(.5, color='red', alpha=0.5, lw=10)  # median position
    ax.set_facecolor('#E6E6E6')
    plt.title(str(player) + '\nFB Percentile Chart', weight='bold', size=24)
    st.pyplot(plt)


    playerdf = totaldf[totaldf['Player'] == (player)]
    st.dataframe(playerdf)

    st.text('')
    st.title('FBs with Avg Percentile Greater Than 0.75')
    st.table(pdf)



if __name__ == "__main__":
    main()

#st.set_option('server.enableCORS', True)
# to run : streamlit run "/Users/michael/Documents/Python/Codes/Scandinavia Player Search App.py"
