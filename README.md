# Self-Improving Language Model Efficiency

We train **Qwen2.5-1.5B** with **GRPO** (a reinforcement-learning method) and test whether choosing practice problems by difficulty lets the model improve itself using **fewer GPU-hours**, without losing accuracy.

- **Datasets:** [Orca-Math](https://huggingface.co/datasets/microsoft/orca-math-word-problems-200k) (math word problems) and [calculus-dataset](https://huggingface.co/datasets/di-zhang-fdu/calculus-dataset) (symbolic calculus)
- **Four strategies compared:** random, filtering, target difficulty, easy-to-hard
- **Tech stack:** Python 3.11, PyTorch, Hugging Face (TRL, transformers, datasets), Unsloth, vLLM, SymPy, Weights & Biases

Laptops are for writing code and preparing data. **All training runs on Kaggle or Colab GPUs.**

---

## Setup

### Windows (PowerShell)

```powershell
# 1. Install Python 3.11 (once)
py install 3.11

# 2. Get the code
git clone https://github.com/aartigapo/Self-Improving-Language-Model-Efficiency.git
cd Self-Improving-Language-Model-Efficiency

# 3. Create and activate the environment
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1

# 4. Install packages
pip install -r requirements-local.txt
```

If activation says "running scripts is disabled", run `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned`, answer `Y`, and try again.

### Mac (Terminal)

```bash
# 1. Install Git (once) — click Install in the pop-up
xcode-select --install
# Install Python 3.11 from python.org (macOS universal2 installer)

# 2. Get the code
git clone https://github.com/aartigapo/Self-Improving-Language-Model-Efficiency.git
cd Self-Improving-Language-Model-Efficiency

# 3. Create and activate the environment
python3.11 -m venv .venv
source .venv/bin/activate

# 4. Install packages
pip install -r requirements-local.txt
```

### Check it worked

Your terminal should start with `(.venv)`. Then run:

```
python --version
python -c "from sympy.parsing.latex import parse_latex; print(parse_latex(r'\frac{-7}{(x+4)^{2}}'))"
```

You should see `Python 3.11.x` and `-7/(x + 4)**2`.

**Each time you come back:** open the project folder and activate the environment again (step 3).

**VS Code:** open the project folder, install the **Python** extension by Microsoft, then **Ctrl/Cmd + Shift + P → Python: Select Interpreter → `.venv`**.

---

## Branches

```
your task branch  →  dev  →  main
```

- **`main`**: tested, working version. Our safe backup. Only updated from `dev` at milestones.
- **`dev`**: where everyone's finished work is combined. If something clashes, it happens here, not on `main`.
- **Task branches**: one per task, named `type/short-description` using the commit types below, e.g. `feat/sympy-checker`, `chore/kaggle-setup`, `docs/progress-report`.

**Starting a task:** always branch from the latest `dev`.

```
git checkout dev
git pull origin dev
git checkout -b feat/sympy-checker
git push -u origin feat/sympy-checker
```

**Daily workflow:**

```
git checkout <your-branch>
git pull origin dev                 # get teammates' latest work
# ...do your work...
git status                          # .venv must NOT appear
git add <files>
git commit -m "feat(data): describe what changed"
git push
```

Then on GitHub, open a **pull request into `dev`** and request a review from every other team member.

---

## How we split work

All three of us work on the same stage each week, splitting its tasks three ways, so everyone works on data, training and analysis. Weekly tasks, branches and "done when" checks are in the team planner. We meet every Thursday to review progress and assign the next week's tasks.

---

## Commit messages

We follow [Conventional Commits](https://www.conventionalcommits.org):

---

## Rules

1. **Never commit directly to `main` or `dev`.** Use your task branch and a pull request.
2. **Every pull request must be approved by all other team members** before it is merged.
3. **Pull `dev` before you start working** each day.
4. **Never commit secrets** (Hugging Face, W&B or GitHub tokens). Keep them in `.env` or Kaggle/Colab Secrets.
5. **Never commit datasets or model files.** They go on the Hugging Face Hub.
6. **Never train on validation or test data.**
7. **Only the selection strategy changes between experiment runs.** Same model, data, settings and GPU type.
8. **Don't upgrade library versions** without telling the team.

---

## More Help

- Ask in the group chat before changing anything shared.
- Start here for GRPO: the *DeepSeekMath* paper (arXiv:2402.03300), GRPO section.
- Training library docs: https://github.com/huggingface/trl