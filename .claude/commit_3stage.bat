@echo off
REM Commit A: 3-stage restructure

cd "C:\Users\Bert Berkers\PycharmProjects\UrbanRepML"

echo Staging renamed directories and new stage3_analysis...
git add stage1_modalities/ stage2_fusion/ stage3_analysis/

echo Staging deletions of old paths...
git add -u modalities/ urban_embedding/

echo Staging configuration and documentation updates...
git add pyproject.toml CLAUDE.md README.md
git add configs/netherlands_pipeline.yaml
git add examples/
git add scripts/analysis/ scripts/netherlands/ scripts/tools/
git add .gitignore

echo Creating Commit A: 3-stage restructure...
git commit -m "$(cat <<'EOF'
refactor: restructure into explicit 3-stage pipeline

Rename modalities/ → stage1_modalities/, urban_embedding/ → stage2_fusion/.
Create stage3_analysis/ from stage2_fusion/analysis/.
Delete visualize_clusters_simple.py (superseded).
Update all imports, pyproject.toml, CLAUDE.md, README.md.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
EOF
)"

echo.
echo Commit A complete.
echo.

echo Staging remaining files for Commit B...
git add -u study_areas/
git add .claude/
git add specs/
git add .python-version .mcp.json
git add scripts/alphaearth_earthengine_retrieval/ scripts/archive/ scripts/preprocessing_auxiliary_data/ scripts/processing_modalities/
git add scripts/netherlands/apply_pca_alphaearth.py scripts/netherlands/debug_cone_data.py scripts/netherlands/infer_cone_alphaearth.py scripts/netherlands/run_netherlands_lattice_unet.py scripts/netherlands/test_cone_forward_pass.py scripts/netherlands/train_cone_alphaearth.py scripts/netherlands/train_lattice_unet_res10_cones.py
git add data/study_areas/

echo Creating Commit B: infrastructure and cascadia cleanup...
git commit -m "$(cat <<'EOF'
chore: add project infrastructure and clean up cascadia archive data

Add .claude/ agent definitions, specs/, .python-version, .mcp.json.
Add remaining scripts and study area configs.
Remove cascadia archive intermediates (superseded).

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
EOF
)"

echo.
echo Commit B complete.
echo.
echo Showing last 8 commits...
git log --oneline -8

echo.
echo Showing remaining unstaged files (should be minimal)...
git status --short | head -20

echo.
echo All done!
