# LogNub

[![BuildAndTest](https://github.com/ChethanUK/lognub/actions/workflows/build_test.yml/badge.svg)](https://github.com/ChethanUK/lognub/actions/workflows/build_test.yml) [![PreCommitChecks](https://github.com/ChethanUK/lognub/actions/workflows/code_quality_lint_checkers.yml/badge.svg)](https://github.com/ChethanUK/lognub/actions/workflows/code_quality_lint_checkers.yml) [![CodeQL](https://github.com/ChethanUK/lognub/actions/workflows/codeql-analysis.yml/badge.svg)](https://github.com/ChethanUK/lognub/actions/workflows/codeql-analysis.yml) [![codecov](https://codecov.io/gh/ChethanUK/lognub/branch/master/graph/badge.svg?token=HRI9hoE5ru)](https://codecov.io/gh/ChethanUK/lognub)

Dumb Loguru Wrap Package extracted from different internal package

NOTE: It's just curated stuff, Created for personal usage.

## TODO

1. Move logwrap [on top of loguru] extension out as a seperate package.
1. Add Test containers for [amundsen](https://www.amundsen.io/amundsen/), etc..

## Getting Started

1. Setup [SDKMAN](#setup-sdkman)
1. Setup [Java](#setup-java)
1. Setup [Apache Spark](#setup-apache-spark)
1. Install [Poetry](#poetry)
1. Install Pre-commit and [follow instruction in here](PreCommit.MD)
1. Run [tests locally](#running-tests-locally)

### Setup SDKMAN

SDKMAN is a tool for managing parallel Versions of multiple Software Development Kits on any Unix based
system. It provides a convenient command line interface for installing, switching, removing and listing
Candidates. SDKMAN! installs smoothly on Mac OSX, Linux, WSL, Cygwin, etc... Support Bash and ZSH shells. See
documentation on the [SDKMAN! website](https://sdkman.io).

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

Install the Java 8:
$ sdk install java 8.0.292.hs-adpt

Set Java 8 as default Java Version:
$ sdk default java 8.0.292.hs-adpt

OR 

To install For Java 11
$ sdk install java 11.0.10.hs-adpt
```

### Setup Apache Spark

Install Java Now open favourite terminal and enter the following:

```bash
List the Apache Spark versions:
$ sdk list spark

To install For Spark 3
$ sdk install spark 3.0.2

To install For Spark 3.1
$ sdk install spark 3.0.2
```

## Install PyEnv and Python 3.8

Either install pyenv via brew or github:
```bash
brew install pyenv

Then setup in zshrc:
echo 'eval "$(pyenv init --path)"' >> ~/.zprofile

echo 'eval "$(pyenv init -)"' >> ~/.zshrc
```

Then Install Python 3.8:

```bash
pyenv install 3.8.11
```

cd allocator project directory:
```bash
# Checkout git repo of iAllocator
cd dse-iAllocator 
cd customer/allocator
# Now set default python version as 3.8.11
pyenv local 3.8.11
# Verify python version
python -V
Python 3.8.11
```

## Create VirtualENV

Create local venv inside allocator project:
```bash
cd dse-iAllocator [Repo directory] 
cd customer/allocator
# Create virtualenv locally 
python -m venv .venv
# Verify activate file exists 
ls -G .venv/bin
# Activate virtual env python
source .venv/bin/activate
```

Verify virtual env and python is working and .venv is activated:

```bash
which python
# should end with {$ROOT_DIR}.venv/bin/python
which pip
# should end with {$ROOT_DIR}.venv/bin/pip
```

### Poetry

Poetry [Commands](https://python-poetry.org/docs/cli/#search) - Python package management tool

Install Poetry:

Install poetry using brew
```bash
brew install poetry
```

OR

Follow instructions for Linux: [here](https://python-poetry.org/docs/#osx--linux--bashonwindows-install-instructions)

```bash
# For osx / linux / bash on windows install:
curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python -
```

Install the dep packages:

NOTE: Make sure you are connected to OpenVPN[since some are internal packages - catalog_client] 
```bash
Install psycopg2 binary via PIP:
$ pip install psycopg2-binary==2.9.1

Install rest of packages via Poetry:

$ poetry install

# --tree: List the dependencies as a tree.
# --latest (-l): Show the latest version.
# --outdated (-o): Show the latest version but only for packages that are outdated.
poetry show -o

To update any package:
#$ poetry update pandas
```

## Running Tests Locally

Take a look at tests in `tests/dataquality` and `tests/jobs`

```bash
$ poetry run pytest
Ran 95 tests in 96.95s
```

Thats it, ENV is setup
