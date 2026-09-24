# GitHub upload instructions

Recommended repository name: `apsp-knn-benchmark`

Recommended description:

> Reproduction code, benchmark outputs, and Colab notebooks for APSP performance analysis on k-nearest-neighbor graphs using SciPy.

## Option A — Git command line

```bash
git init
git add .
git commit -m "Initial public reproduction package"
git branch -M main
git remote add origin https://github.com/gokhanturan/apsp-knn-benchmark.git
git push -u origin main
```

## Option B — GitHub web upload

1. Create a new public repository named `apsp-knn-benchmark`.
2. Do not add a GitHub-generated README, `.gitignore`, or license during repository creation because this package already contains a README and `.gitignore`.
3. Upload the contents of this folder to the repository root.
4. Confirm that the `notebooks/`, `src/`, `results/`, `graphs/`, `scaling_graphs/`, `data/`, `figures/`, and `docs/` folders are visible.
5. Open each notebook from GitHub and test the **Open in Colab** badge.

## Suggested manuscript availability statement

> The source code, benchmark outputs, graph files, and Colab reproduction notebooks are publicly available at https://github.com/gokhanturan/apsp-knn-benchmark.

After the article is published, add the article DOI to `README.md` and `CITATION.cff`.
