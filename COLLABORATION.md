# Collaboration Guide

## Repository

```text
https://github.com/Maaskk/gnn-optimization-under-graph-perturbations-cora
```

## Invite Your Teammate

Open repository settings, then go to:

```text
Settings > Access > Collaborators
```

Invite your teammate by GitHub username or email. Give them write access so they can push branches and open pull requests.

Direct page:

```text
https://github.com/Maaskk/gnn-optimization-under-graph-perturbations-cora/settings/access
```

## Clone the Project

```bash
git clone https://github.com/Maaskk/gnn-optimization-under-graph-perturbations-cora.git
cd gnn-optimization-under-graph-perturbations-cora
```

## Branches

Ossama works on:

```bash
git checkout ossama/baseline
```

Teammate works on:

```bash
git checkout teammate/perturbations
```

## Daily Workflow

Before starting work:

```bash
git pull
```

After making changes:

```bash
git status
git add .
git commit -m "short clear message"
git push
```

## Merge Workflow

1. Push your branch.
2. Open a pull request into `main`.
3. Ask the other person to review.
4. Merge only after the code runs and results are saved correctly.

