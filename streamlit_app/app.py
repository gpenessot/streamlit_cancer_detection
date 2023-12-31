from collections import OrderedDict
import streamlit as st
import os
# TODO : change TITLE, TEAM_MEMBERS and PROMOTION values in config.py.
import config

# TODO : you can (and should) rename and add tabs in the ./tabs folder, and import them here.
from tabs import intro, cancer, donnees, modele, results2, bilan

STREAMLIT_CLOUD_ROOT_PATH='/app/streamlit_cancer_detection/streamlit_app/'

st.set_page_config(
    page_title=config.TITLE,
    page_icon="https://datascientest.com/wp-content/uploads/2020/03/cropped-favicon-datascientest-1-32x32.png",
)

with open(os.path.join(STREAMLIT_CLOUD_ROOT_PATH, "style.css"), "r") as f:
    style = f.read()

st.markdown(f"<style>{style}</style>", unsafe_allow_html=True)


# TODO: add new and/or renamed tab in this ordered dict by
# passing the name in the sidebar as key and the imported tab
# as value as follow :
TABS = OrderedDict(
    [
        (intro.sidebar_name, intro),
        (cancer.sidebar_name, cancer),
        (donnees.sidebar_name, donnees  ),
        (modele.sidebar_name, modele),
#        (results.sidebar_name, results),
        (results2.sidebar_name, results2),
        (bilan.sidebar_name, bilan),
    ]
)


def run():
    st.sidebar.image(os.path.join(STREAMLIT_CLOUD_ROOT_PATH, "assets/logo_blanc.png")) #, width=100
    tab_name = st.sidebar.radio("", list(TABS.keys()), 0)
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"## {config.PROMOTION}")
    st.sidebar.markdown("## Membres de l'équipe :")
    for member in config.TEAM_MEMBERS:
        st.sidebar.markdown(member.sidebar_markdown(), unsafe_allow_html=True)

    tab = TABS[tab_name]

    tab.run()


if __name__ == "__main__":
    run()
