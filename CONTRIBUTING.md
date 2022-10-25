# Welcome to OasisLMF contributing guide :wave:

Thank you for investing your time in contributing to this project!

<!-- Read our [Code of Conduct](./CODE_OF_CONDUCT.md) to keep our community approachable and respectable. -->

In this guide you will get an overview of the coding style that we follow and of the contribution workflow from opening an issue, creating a pull request (PR), reviewing, and merging the PR.


## Coding principles

The goal of the these coding guidelines is to ensure that the OasisLMF repositories are highly maintainable, well documented, and well structured.

Please adhere to the following principles when contributing to the code base:

 - be **DRY** (aka Don't Repeat Yourself): avoid code duplication as much as possible. If an element (variable, constant, data type, function, etc.) is already defined somewhere else in the package, then re-use or augment the existing definition.
 
 - add proper **documentation** to the code:
   - on general terms, we follow the Google Python Style Guide for comments and docstrings: see [Chapter 3.8](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings).
   - all functions must have a docstring describing their purpose, the data type and the content of all input and output variables. Where external results (e.g., specific algorithms) are used, a short note or reference to the external source should be included.
   - the docstrings need to be concise and essential, yet complete.
   - in functions implementing non-trivial logic and/or complex algorithms, the code should be annotated with short and informative comments making clear the logic and the flow, and the reasoning behind non-obvious implementation decisions.
   
  - format the code to make it **PEP8 compliant**. This is easily done with tools like `autopep8`: see how to use formatters in [VS code](https://code.visualstudio.com/docs/python/editing#_formatting) and [PyCharm](https://www.jetbrains.com/help/pycharm/reformat-and-rearrange-code.html). 
  
When in doubt about the style, have a look at the [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html): it contains PEP-compliant style recommendations and examples.

Code that doesn't follow the above principles shall not be merged into the codebase.

## Getting started

### Issues

**Create a new issue:** if you spot a problem with the code or the docs, search if an issue already exists in that repository. For example, for the OasisLMF/OasisLMF repository, see the [existing issues here](https://github.com/OasisLMF/OasisLMF/issues). If a related issue doesn't exist, you can open a new issue. 

**Solve an issue:** scan through our existing issues to find one that interests you. You can narrow down the search using `labels` as filters. If you find an issue to work on, you are welcome to open a PR with a fix.

### Make Changes

1. [Install `git`](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git).

2. Fork the repository.
- Using GitHub Desktop:
  - [Getting started with GitHub Desktop](https://docs.github.com/en/desktop/installing-and-configuring-github-desktop/getting-started-with-github-desktop) will guide you through setting up Desktop.
  - Once Desktop is set up, you can use it to [fork the repo](https://docs.github.com/en/desktop/contributing-and-collaborating-using-github-desktop/cloning-and-forking-repositories-from-github-desktop)!

- Using the command line:
  - [Fork the repo](https://docs.github.com/en/github/getting-started-with-github/fork-a-repo#fork-an-example-repository) so that you can make your changes without affecting the original project until you're ready to merge them.

3. Create a working branch and start with your changes!

### Commit your update

Commit and push the changes once you are happy with them.

### Pull Request

When you're finished with the changes, create a pull request, also known as a PR.
- Fill the PR template in all its parts.
- As a general rule, the **develop** branch is the target of all PRs for new features or bugfixes.
- Add the "Ready for Review" label to this PR.
- Don't forget to [link PR to issue](https://docs.github.com/en/issues/tracking-your-work-with-issues/linking-a-pull-request-to-an-issue) if you are solving one.
- Enable the checkbox to [allow maintainer edits](https://docs.github.com/en/github/collaborating-with-issues-and-pull-requests/allowing-changes-to-a-pull-request-branch-created-from-a-fork) so the branch can be updated for a merge.
Once you submit your PR, a Docs team member will review your proposal. We may ask questions or request additional information.
- We may ask for changes to be made before a PR can be merged, either using [suggested changes](https://docs.github.com/en/github/collaborating-with-issues-and-pull-requests/incorporating-feedback-in-your-pull-request) or pull request comments. You can apply suggested changes directly through the UI. You can make any other changes in your fork, then commit them to your branch. In many cases, we can help by implementing the required changes ourselves.
- As you update your PR and apply changes, mark each conversation as [resolved](https://docs.github.com/en/github/collaborating-with-issues-and-pull-requests/commenting-on-a-pull-request#resolving-conversations).
- If you run into any merge issues, checkout this [git tutorial](https://github.com/skills/resolve-merge-conflicts) to help you resolve merge conflicts and other issues.

### Your PR is merged!

Congratulations :tada::tada: The OasisLMF team thanks you :sparkles:. 

