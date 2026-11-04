#!/usr/bin/env bash
# Install a pre-commit hook that refuses to commit SOEP micro-data.
#
# .gitignore already excludes these paths, but `git add -f` overrides it and a
# committed file is hard to remove from history afterwards. This hook is the
# second line of defence. Run once after cloning:
#
#     bash scripts/install_git_hooks.sh
 
set -euo pipefail
 
REPO_ROOT="$(git rev-parse --show-toplevel)"
HOOK="${REPO_ROOT}/.git/hooks/pre-commit"
 
cat > "${HOOK}" <<'HOOK_EOF'
#!/usr/bin/env bash
# Blocks SOEP micro-data from entering the repository.
set -euo pipefail
 
STAGED=$(git diff --cached --name-only --diff-filter=ACM)
[ -z "${STAGED}" ] && exit 0
 
BLOCKED=""
 
# 1. Statistical data formats anywhere in the tree.
while IFS= read -r f; do
    case "${f}" in
        *.dta|*.sav|*.zsav|*.por|*.sas7bdat|*.xpt|*.rds|*.RData|*.rda)
            BLOCKED="${BLOCKED}\n  ${f}  (statistical data format)" ;;
        *.parquet|*.feather|*.h5|*.hdf5|*.pkl|*.pickle)
            BLOCKED="${BLOCKED}\n  ${f}  (binary data format)" ;;
    esac
done <<< "${STAGED}"
 
# 2. Anything under a dataset data/ or person-level output directory.
while IFS= read -r f; do
    case "${f}" in
        datasets/*/data/*)
            [ "$(basename "${f}")" = ".gitkeep" ] || \
                BLOCKED="${BLOCKED}\n  ${f}  (dataset data directory)" ;;
        datasets/*/outputs/simulated_data/*|datasets/*/outputs/personas/*|datasets/*/outputs/models/*|datasets/*/outputs/logs/*)
            [ "$(basename "${f}")" = ".gitkeep" ] || \
                BLOCKED="${BLOCKED}\n  ${f}  (contains individual responses)" ;;
    esac
done <<< "${STAGED}"
 
# 3. Any CSV/JSONL outside the explicitly allowed locations.
# Keep this list in sync with the allow-list at the end of .gitignore
# (docs/, outputs/tables/, tests/fixtures/, datasets/*/outputs/results/).
while IFS= read -r f; do
    case "${f}" in
        docs/*|tests/fixtures/*|outputs/tables/*|datasets/*/outputs/results/*) continue ;;
        *.csv|*.tsv|*.jsonl)
            BLOCKED="${BLOCKED}\n  ${f}  (tabular file outside the allowed aggregate-output locations)" ;;
    esac
done <<< "${STAGED}"
 
if [ -n "${BLOCKED}" ]; then
    echo ""
    echo "COMMIT BLOCKED — these staged files look like survey micro-data:"
    echo -e "${BLOCKED}"
    echo ""
    echo "The SOEP data use agreement with DIW Berlin forbids passing the data to"
    echo "third parties. That includes private repositories and any cloud-hosted"
    echo "git remote."
    echo ""
    echo "If a file is genuinely aggregate and safe to publish, commit it with:"
    echo "    git commit --no-verify"
    echo "and say in the commit message why it is safe."
    echo ""
    exit 1
fi
 
exit 0
HOOK_EOF
 
chmod +x "${HOOK}"
echo "Installed pre-commit hook at ${HOOK}"
echo "Test it with:  git add -f some.dta && git commit -m test"