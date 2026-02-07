# PowerShell script to complete 3-stage restructure commits

Set-Location "C:\Users\Bert Berkers\PycharmProjects\UrbanRepML"

Write-Host "=== Step 1: Delete stage2_fusion/analysis/ directory ===" -ForegroundColor Cyan
if (Test-Path "stage2_fusion\analysis") {
    Remove-Item -Recurse -Force "stage2_fusion\analysis"
    Write-Host "Deleted stage2_fusion/analysis/" -ForegroundColor Green
}

Write-Host "`n=== Step 2: Delete visualize_clusters_simple.py ===" -ForegroundColor Cyan
if (Test-Path "stage1_modalities\alphaearth\visualize_clusters_simple.py") {
    Remove-Item -Force "stage1_modalities\alphaearth\visualize_clusters_simple.py"
    Write-Host "Deleted visualize_clusters_simple.py" -ForegroundColor Green
}

Write-Host "`n=== COMMIT A: 3-stage restructure ===" -ForegroundColor Yellow
Write-Host "Staging renamed directories and new stage3_analysis..."
git add stage1_modalities/ stage2_fusion/ stage3_analysis/

Write-Host "Staging deletions of old paths..."
git add -u modalities/ urban_embedding/

Write-Host "Staging configuration and documentation updates..."
git add pyproject.toml CLAUDE.md README.md
git add configs/netherlands_pipeline.yaml
git add examples/
git add scripts/analysis/ scripts/netherlands/ scripts/tools/
git add .gitignore

Write-Host "`nCreating Commit A..." -ForegroundColor Green
git commit -F .claude\commit_msg_a.txt

Write-Host "`n=== COMMIT B: infrastructure and cascadia cleanup ===" -ForegroundColor Yellow
Write-Host "Staging remaining files..."
git add -u study_areas/
git add .claude/
git add specs/
git add .python-version .mcp.json 2>$null
git add scripts/alphaearth_earthengine_retrieval/ scripts/archive/ scripts/preprocessing_auxiliary_data/ scripts/processing_modalities/ 2>$null
git add "scripts/netherlands/apply_pca_alphaearth.py" "scripts/netherlands/debug_cone_data.py" "scripts/netherlands/infer_cone_alphaearth.py" "scripts/netherlands/run_netherlands_lattice_unet.py" "scripts/netherlands/test_cone_forward_pass.py" "scripts/netherlands/train_cone_alphaearth.py" "scripts/netherlands/train_lattice_unet_res10_cones.py" 2>$null
git add data/study_areas/ 2>$null

Write-Host "`nCreating Commit B..." -ForegroundColor Green
git commit -F .claude\commit_msg_b.txt

Write-Host "`n=== VERIFICATION ===" -ForegroundColor Cyan
Write-Host "`nLast 8 commits:"
git log --oneline -8

Write-Host "`nRemaining status (should be minimal):"
git status --short | Select-Object -First 20
