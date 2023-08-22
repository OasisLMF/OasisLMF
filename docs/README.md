How to work on/build the docs
===

Requirements
---
Install the following packages in your local Python environment:
```
pip install sphinx sphinx-autoapi 

```
Also, make sure to have `ods_tools` installed from the development branch in order to include the latest features:
```sh
pip install git+https://git@github.com/OasisLMF/ODS_Tools
```

Context
-------
The documentation is set up to use `sphinx-autoapi`, which is a wrapper around `sphinx-build`
that builds documentation for the entire `oasislmf` package, automatically navigating through the
modules and sub-modules recursively.

To build the `oasislmf` documentation, run, from the repository root directory:
```sh
./docs/build.sh
```
The HTML documentation will be created in `docs/build/html/`.

By opening `docs/build/html/index.html` with a browser, it will be possible to navigate the documentation locally.


Automation
----------
Upon committing and pushing the docs to the GitHub repository, the documentation will be:
 - built for all PRs targeting `main`, with the HTML documentation being stored as a downloadable artifact (a `github-pages.zip` file);
 - built and deployed to GitHub pages (i.e., stored in the `gh-pages` branch of the repository) for all push commits on the `main` branch.



