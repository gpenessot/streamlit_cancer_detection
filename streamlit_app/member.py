import os
PATH = os.getcwd()
print(PATH)

class Member:
    def __init__(
        self, name: str, linkedin_url: str = None, github_url: str = None
    ) -> None:
        self.name = name
        self.linkedin_url = linkedin_url
        self.github_url = github_url

    def sidebar_markdown(self):

        markdown = f'<b style="display: inline-block; color: white; vertical-align: middle; height: 100%"> ▹ {self.name}</b>'

        if self.linkedin_url is not None:
            markdown += f' <a href={self.linkedin_url} target="_blank"><img src="https://raw.githubusercontent.com/ecoudron/hello-world/master/images/linkedin-logo-black.png" alt="linkedin" width="25" style="vertical-align: middle; margin-left: 5px"/></a> '

        if self.github_url is not None:
            markdown += f' <a href={self.github_url} target="_blank"><img src="https://raw.githubusercontent.com/ecoudron/hello-world/master/images/github-logo.png" alt="github" width="20" style="vertical-align: middle; margin-left: 5px"/></a> '

        return markdown
