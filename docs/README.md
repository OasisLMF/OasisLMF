How to work on/build the docs
===

Requirements
---
Install the following packages in your local Python environment:
```
pip install sphinx sphinx-multiversion 

```
Also, make sure to have `ods_tools` installed from the development branch in order to include the latest features:
```sh
pip install git+https://git@github.com/OasisLMF/ODS_Tools
```

Context
---
The documentation is set up to use `sphinx-multiversion`, which is a wrapper around `sphinx-build`
that automatically builds documentation for all release tags and branches whitelisted in `conf.py` settings
(see variables starting with `smv_`). This is useful as it automatically populates a **Version** tab in the bottom-left of the docs webpages to navigate through different versions of the documentation.

To build the multi-version documentation:
```sh
cd docs
sphinx-multiversion source build/html
```
where `source/` is the directory containing all the documentation and `build/html/` is the target directory where the rendered documentation is to be stored. For each version of the documentation, there will be a directory inside the target directory, containing the rendered documentation.

For example, to produce the documentation for the `master` and `develop` branches, we set `smv_branch_whitelist = r'^(master|develop)'` in `conf.py`. Then, we run `sphinx-multiversion source build/html`. This will produce the following directories:
```sh
build
└── html
    ├── develop
    └── master
```

By opening, say, `build/html/develop/index.html`, it will be possible to navigate to the documentation of the `master` branch by using the links in the **Version** tab.

Important note
---
`sphinx-multiversion` produces documentation by running a `git checkout` of the desired release tags and branches, whitelisted in `conf.py` as shown above. Therefore, when working locally, all changes to the docs will not be reflected in the documentation produced by `sphinx-multiversion` until they are committed.

When developing the docs locally, it is thus faster to use `sphinx-build` which will produce the documentation using the files currently checked out, including the latest, uncommitted, changes:
```sh
cd docs
sphinx-build source build/html
```

[WORK IN PROGRESS] Upon committing and pushing the docs to the GitHub repository, the documentation will be built and deployed using `sphinx-multiversion` using a GitHub action.



