# LogNub

[![BuildAndTest](https://github.com/ChethanUK/lognub/actions/workflows/build_test.yml/badge.svg)](https://github.com/ChethanUK/lognub/actions/workflows/build_test.yml) [![PreCommitChecks](https://github.com/ChethanUK/lognub/actions/workflows/code_quality_lint_checkers.yml/badge.svg)](https://github.com/ChethanUK/lognub/actions/workflows/code_quality_lint_checkers.yml) [![CodeQL](https://github.com/ChethanUK/lognub/actions/workflows/codeql-analysis.yml/badge.svg)](https://github.com/ChethanUK/lognub/actions/workflows/codeql-analysis.yml) [![codecov](https://codecov.io/gh/ChethanUK/lognub/branch/master/graph/badge.svg?token=HRI9hoE5ru)](https://codecov.io/gh/ChethanUK/lognub)

Dumb Loguru Wrap Package extracted from different internal package

NOTE: It's just curated stuff, Created for personal usage.

## TODO

1. Move logwrap [on top of loguru] extension out as a seperate package.
1. Add Test containers for [amundsen](https://www.amundsen.io/amundsen/), etc..

## Getting Started

1. Setup [SDKMAN](#setup-sdkman)
1. Install [Poetry](#poetry)
1. Install Pre-commit and [follow instruction in here](PreCommit.MD)
1. Run [tests locally](#running-tests-locally)

### Setup SDKMAN

SDKMAN is a tool for managing parallel Versions of multiple Software Development Kits on any Unix based system. It provides a convenient command line interface for installing, switching, removing and listing Candidates.
SDKMAN! installs smoothly on Mac OSX, Linux, WSL, Cygwin, etc... Support Bash and ZSH shells.
See documentation on the [SDKMAN! website](https://sdkman.io).

Open your favourite terminal and enter the following:

```bash
$ curl -s https://get.sdkman.io | bash
If the environment needs tweaking for SDKMAN to be installed,
the installer will prompt you accordingly and ask you to restart.

Next, open a new terminal or enter:

$ source "$HOME/.sdkman/bin/sdkman-init.sh"

Lastly, run the following code snippet to ensure that installation succeeded:

$ sdk version
```

### Setup Java

Install Java Now open favourite terminal and enter the following:

```bash
List the AdoptOpenJDK OpenJDK versions
$ sdk list java

To install For Java 11
$ sdk install java 11.0.10.hs-adpt

To install For Java 11
$ sdk install java 8.0.292.hs-adpt
```

### Poetry

Poetry [Commands](https://python-poetry.org/docs/cli/#search)

```bash
poetry install

poetry update

# --tree: List the dependencies as a tree.
# --latest (-l): Show the latest version.
# --outdated (-o): Show the latest version but only for packages that are outdated.
poetry show -o
```

## Running Tests Locally

Take a look at tests in `tests/dataquality` and `tests/jobs`

```bash
$ poetry run pytest
Ran 95 tests in 96.95s
```
