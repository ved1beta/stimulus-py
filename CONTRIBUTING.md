# Contributing

Contributions are most welcome! Follow the stepts below to get started.

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

## Understanding the codebase

Unfortunately, at this stage, we have no tutorials ready or in-depth README. So you will have to dive into the codebase by yourself. 
Thankfully, every function or class has a docstring that should give you a good idea of what it does, and docs are building (you can run `make docs` to access dev branch docs which should be more up to date than the main branch docs).

After running `make setup`, you should have a `.venvs` directory in the root of the repository.

To activate it, run:

```bash
source .venvs/<python_version>/bin/activate
```

stimulus should now be installed as a librairy!

Then, you can install extra dependencies (for instance jupyter) using `uv pip install <package>`.

We recommand you install jupyter, spin up a notebook and take the titanic datasets formated to the stimulus format in `tests/test_data/titanic/titanic_stimulus.csv` as well as the model in `tests/test_model/titanic_model.py`. 

The configurations files (stimulus uses those to generate different versions of the model and datasets) are in `tests/test_data/titanic/titanic.yaml` and `tests/test_model/titanic_model_{gpu/cpu}.yaml`. 

From there, you should first try to load the dataset in a data_handler class (see `stimulus.data.data_handler.py`). For this, you will need loader classes (see `stimulus.data.loaders.py`). Currently, loaders use a config system, the split config in `tests/test_data/titanic/titanic_sub_config.yaml` should be enough to get you started. 

Then, you can try to load the dataset in the pytorch dataset utility (see `stimulus.data.handlertorch.py`). And run the model on a few examples to see what is going on. 

Finally, you can call tuning scripts (see `stimulus.learner.raytune_learner.py`) - make sure you import ray and call ray.init() first - and run a hyper parameter tuning run on the model using the model config `tests/test_model/titanic_model_{gpu/cpu}.yaml`. 

Since this is not so well documented, it is possible you encounter issues, bugs, things that are weird to you, unintuitive, etc. I will be here to answer questions, either on nf-core slack or on discord. Those questions are extremely valuable to get the documentation up to speed.

## Contributing

At this stage, you will mostly be understanding the codebase and have your own ideas on how to improve it. If so, please make an issue and discuss on discord/slack. Otherwise, you can pick one of the many issues already opened and work from there. 

### Things that are always welcome (especially interesting for newcomers):

- Improving documentation:

This is the most impactful thing you can do as a newcomer (since you get to write the documentation you wish you had when you started) + it will help you understand the codebase better. PR's aiming to improve documentation are also very easy to review and accept, those will be prioritized.

- Building tutorials: 

Now that you understand the codebase a bit better, helping others understanding it as well will always be extremely valuable, and as with documentation PR's, those will be prioritized.

- Adding Encoders/Transforms/Splitters

This librairy lives and dies by the number of encoders/transforms/splitters offered, so adding those will always improve the software.

- Quality of life improvements

As users, you have a much better understanding of the pain points than we do. If you have ideas on how to improve the software, please share them!

- Bug fixes 

- Performance improvements 

### Things that are always welcome discussing but not necessarily easy to get for newcomers: 

- Refactoring 

Often refactoring code is needed to improve the codebase, make it more readable or flexible (and refactoring making the codebase more readable will always be highly valued). 

- Porting stimulus to non-bio fields

This will sometime require extensive refactor and good understanding of the codebase to make sure it does not break anything.

- Adding extra functionality (specifically downstream analysis, interpretation methods etc...)

Stimulus would be a lot more useful if it could perform downstream model analysis, interpretability methods, overfitting analysis etc.. All of those things are on the roadmap, but I guess the codebase need to be well understood first (however, raising issues to discuss how to do this is always welcome).

## How to contribute code

### First thing you should do

Fork the repository, then : 

1. create a new branch: `git switch -c feature-or-bugfix-name`
1. edit the code and/or the documentation

### Commit guidelines : 

Please write atomic commits, meaning that each commit should be a single change. This is good practice since it allows everybody to review much faster!

When we push a new release, we use `make changelog` to generate the changelog. This has one caveat, it only works if commits messages are following the angular commit guidelines. Therefore, if you want your contributions to be seen, you need to write your commit messages following [their contributing guide](https://github.com/angular/angular/blob/master/CONTRIBUTING.md#commit-message-format).

Below, the relevant part of the angular contribution guide for commit messages: 

```
<type>(<scope>): <short summary>
  │       │             │
  │       │             └─⫸ Summary in present tense. Not capitalized. No period at the end.
  │       │
  │       └─⫸ Commit Scope: animations|bazel|benchpress|common|compiler|compiler-cli|core|
  │                          elements|forms|http|language-service|localize|platform-browser|
  │                          platform-browser-dynamic|platform-server|router|service-worker|
  │                          upgrade|zone.js|packaging|changelog|docs-infra|migrations|
  │                          devtools
  │
  └─⫸ Commit Type: build|ci|docs|feat|fix|perf|refactor|test
```

The `<type>` and `<summary>` fields are mandatory, the `<scope>` field is optional.
Type
Must be one of the following:

- `build`: Changes that affect the build system or external dependencies (example scopes: gulp, broccoli, npm)
- `ci`: Changes to our CI configuration files and scripts (examples: Github Actions, SauceLabs)
- `docs`: Documentation only changes
- `feat`: A new feature
- `fix`: A bug fix
- `perf`: A code change that improves performance
- `refactor`: A code change that neither fixes a bug nor adds a feature
- `test`: Adding missing tests or correcting existing tests

The scope, while optional, is recommended (typically the name of the file you are working on).

**Before pushing:**

The entry-point to run commands and tasks is the `make` Python script,
located in the `scripts` directory. Try running `make` to show the available commands and tasks.
The *commands* do not need the Python dependencies to be installed,
while the *tasks* do.
The cross-platform tasks are written in Python, thanks to [duty](https://github.com/pawamoy/duty).

If you work in VSCode, we provide
[an action to configure VSCode](https://pawamoy.github.io/copier-uv/work/#vscode-setup)
for the project.

1. run `make format` to auto-format the code
1. run `make check` to check everything (fix any warning)
1. run `make test` to run the tests (fix any issue)
1. if you updated the documentation or the project dependencies:
    1. run `make docs`
    1. go to http://localhost:8000 and check that everything looks good

Then you can pull request and we will review. Make sure you join our [slack](https://nfcore.slack.com/channels/deepmodeloptim) hosted on nf-core or find us on discord to talk and build with us!

Once you have your first PR merged, you will be added to the repository as a contributor and your contributions will be aknowledged!


