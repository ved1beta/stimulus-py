# Contributing

Contributions are welcome, and they are greatly appreciated!
Every little bit helps, and credit will always be given.

## Environment setup

Nothing easier!

Fork and clone the repository, then:

```bash
cd stimulus-py
make setup
```

> NOTE:
> If it fails for some reason,
> you'll need to install
> [uv](https://github.com/astral-sh/uv)
> manually.
>
> You can install it with:
>
> ```bash
> curl -LsSf https://astral.sh/uv/install.sh | sh
> ```
>
> Now you can try running `make setup` again,
> or simply `uv sync`.

You now have the dependencies installed.

Run `make help` to see all the available actions!

## Tasks

The entry-point to run commands and tasks is the `make` Python script,
located in the `scripts` directory. Try running `make` to show the available commands and tasks.
The *commands* do not need the Python dependencies to be installed,
while the *tasks* do.
The cross-platform tasks are written in Python, thanks to [duty](https://github.com/pawamoy/duty).

If you work in VSCode, we provide
[an action to configure VSCode](https://pawamoy.github.io/copier-uv/work/#vscode-setup)
for the project.

## Development

As usual:

1. create a new branch: `git switch -c feature-or-bugfix-name`
1. edit the code and/or the documentation

**Before committing:**

1. run `make format` to auto-format the code
1. run `make check` to check everything (fix any warning)
1. run `make test` to run the tests (fix any issue)
1. if you updated the documentation or the project dependencies:
    1. run `make docs`
    1. go to http://localhost:8000 and check that everything looks good


Then you can pull request and we will review. Make sure you join our [slack](https://nfcore.slack.com/channels/deepmodeloptim) hosted on nf-core to talk and build with us!
