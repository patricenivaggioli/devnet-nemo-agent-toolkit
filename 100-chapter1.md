# Getting Started

Clone the NeMo Agent toolkit repository to your lab environment (or loacal machine).

```bash
git clone -b main https://github.com/NVIDIA/NeMo-Agent-Toolkit.git nemo-agent-toolkit
cd nemo-agent-toolkit
```

Initialize, fetch, and update submodules in the Git repository.

```bash
git submodule update --init --recursive
```

Fetch the data sets by downloading the LFS files.

```bash
sudo apt-get update && sudo apt-get install git-lfs
git lfs install
git lfs fetch
git lfs pull
```

Create a Python environment.

```bash
sudo snap install astral-uv --classic
```

```bash
uv venv --python 3.13 --seed .venv
source .venv/bin/activate
```

Install the NeMo Agent toolkit library. To install the NeMo Agent toolkit library along with all of the optional dependencies. Including developer tools (--all-groups) and all of the dependencies needed for profiling and plugins (--all-extras) in the source repository, run the following:

```bash
uv sync --all-groups --all-extras
```

Verify that you've installed the NeMo Agent toolkit library.

```bash
nat --help
nat --version
```

If the installation succeeded, the nat command will log the help message and its current version.
