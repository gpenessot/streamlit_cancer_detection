"""

Config file for Streamlit App

"""

from member import Member


TITLE = "Classification histopathologie"

TEAM_MEMBERS = [
    Member(
        name="Olivier Bertrand",
        linkedin_url="https://www.linkedin.com/in/olivier-bertrand-4552b41b4/",
        github_url="https://github.com/OlivBertrand",
    ),
    Member(
        name="Etienne Coudron",
        linkedin_url="https://www.linkedin.com/in/etiennecoudron/",
        github_url="https://github.com/ecoudron/",
    ),
    Member(
        name="Guillaume Demonsablon",
        linkedin_url="https://www.linkedin.com/in/guillaumedemonsablon/",
        #github_url="https://github.com/guillaumedemonsablon/",
    ),
    Member(
        name="Gaël Penessot",
        linkedin_url="https://www.linkedin.com/in/gael-penessot/",
        github_url="https://github.com/gpenessot",
    ),
]

PROMOTION = "Promotion Formation continue \
             Data Scientist - Novembre 2021"
