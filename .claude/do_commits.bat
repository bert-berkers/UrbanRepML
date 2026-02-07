@echo off
cd "C:\Users\Bert Berkers\PycharmProjects\UrbanRepML"

echo === COMMIT A: 3-stage restructure ===
echo.

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

echo.
echo Creating Commit A...
git commit -F .claude\commit_msg_a.txt

echo.
echo === COMMIT B: infrastructure and cascadia cleanup ===
echo.

echo Staging remaining files...
git add -u study_areas/
git add .claude/
git add specs/
git add .python-version .mcp.json 2>nul
git add scripts/alphaearth_earthengine_retrieval/ scripts/archive/ scripts/preprocessing_auxiliary_data/ scripts/processing_modalities/ 2>nul
git add scripts/netherlands/apply_pca_alphaearth.py scripts/netherlands/debug_cone_data.py scripts/netherlands/infer_cone_alphaearth.py scripts/netherlands/run_netherlands_lattice_unet.py scripts/netherlands/test_cone_forward_pass.py scripts/netherlands/train_cone_alphaearth.py scripts/netherlands/train_lattice_unet_res10_cones.py 2>nul
git add data/study_areas/ 2>nul

echo.
echo Creating Commit B...
git commit -F .claude\commit_msg_b.txt

echo.
echo === VERIFICATION ===
echo.
echo Last 8 commits:
git log --oneline -8

echo.
echo Remaining status (should be minimal):
git status --short
