# TODO FOR JULES - UrbanRepML Critical Tasks

**CONTEXT**: This document was created after an exploration session with Claude Code. We identified critical security issues and architectural problems that need immediate attention.

## 🚨 PRIORITY 1: SECURITY - Remove Exposed API Keys

**CRITICAL**: There may be hardcoded Google Earth Engine API keys somewhere in the codebase that are currently exposed on GitHub.

### Tasks:
1. **Search for exposed credentials**:
   - Check all Python files for hardcoded API keys, service account JSONs, or authentication strings
   - Pay special attention to:
     - `scripts/processing embeddings/alphaearth/` folder
     - Any files using `ee.Authenticate()` or `ee.Initialize()`
     - JSON files that might contain service account credentials
   
2. **Create secure credential management**:
   ```
   UrbanRepML/
   ├── keys/                    # NEW - Add to .gitignore
   │   ├── .env                # Actual credentials (never commit)
   │   └── .env.example        # Template for others
   ```

3. **Update code to use environment variables**:
   - Install python-dotenv: `pip install python-dotenv`
   - Load credentials from keys/.env
   - Update all Earth Engine authentication to use env variables

4. **Fix Git history**:
   - After removing keys, commit with message: "fix: remove exposed API keys and implement secure credential management"
   - Force push to overwrite history: `git push --force origin master`

5. **Rotate compromised keys**:
   - Go to Google Cloud Console
   - Generate new Earth Engine service account key
   - Update local keys/.env with new credentials

## 📁 PRIORITY 2: Clean Up Study Areas Structure

**ISSUE**: The study_areas/ folder contains many scripts that should be in the scripts/ folder. Per CLAUDE.md, study_areas should only contain data, not processing scripts.

### Current problematic structure:
```
study_areas/
├── cascadia/
│   └── scripts/            # 29 Python files that need to be moved!
│       ├── load_alphaearth.py
│       ├── clustering.py
│       ├── visualizations.py
│       └── ... (many more)
├── netherlands/
│   ├── south_holland_fsi99/
│   │   └── run_experiment.py    # Should be moved
│   └── south_holland_hexagonal/
│       └── run_hexagonal.py     # Should be moved
```

### Tasks:
1. **Create new script organization**:
   ```
   scripts/
   ├── cascadia/               # Move all cascadia scripts here
   ├── netherlands/            # Move all netherlands scripts here
   └── earthengine/           # NEW - For Earth Engine API scripts
   ```

2. **Move files** (preserving functionality):
   - Move all 29 files from `study_areas/cascadia/scripts/` to `scripts/cascadia/`
   - Move experiment runners from netherlands subfolders to `scripts/netherlands/`
   - Update all import paths in moved files

3. **Review and consolidate**:
   - Many visualization scripts appear duplicated
   - Clustering scripts could be consolidated
   - Remove obsolete experimental scripts after review

### Scripts to move from cascadia (for your review):
- **Processing**: load_alphaearth.py, modular_tiff_processor.py, stitch_results.py
- **Analysis**: clustering.py, coastal_clustering.py, quick_clustering.py
- **Visualization**: Multiple viz scripts (need consolidation)
- **Utilities**: benchmark_processors.py, check_progress.py, monitor_modular_progress.py
- **Experiments**: spatial_smoothing.py, targeted_gap_elimination.py

## 🌍 PRIORITY 3: Create Earth Engine API Integration

**NEED**: A proper script to fetch AlphaEarth embeddings via Earth Engine API instead of relying on manual downloads.

### Create: `scripts/earthengine/fetch_alphaearth_embeddings.py`

Features needed:
1. **Authentication**:
   - Load service account from keys/.env
   - Initialize Earth Engine with proper credentials

2. **Study area support**:
   - Accept study area name as parameter
   - Load boundary from study_areas/{name}/area_gdf/
   - Create Earth Engine geometry from boundary

3. **AlphaEarth embedding retrieval**:
   - Connect to AlphaEarth dataset on Earth Engine
   - Request embeddings for study area
   - Handle large areas with tiling if needed

4. **Export options**:
   - Export to Google Drive (for manual download)
   - Direct download if file size permits
   - Save to study_areas/{name}/embeddings/alphaearth/

5. **Progress tracking**:
   - Monitor export tasks
   - Provide status updates
   - Handle errors gracefully

## 🏗️ PRIORITY 4: Architecture Alignment

Ensure everything follows the CLAUDE.md principles:

### Key principles to enforce:
1. **SRAI everywhere**: All H3 operations must use SRAI, never h3-py directly
2. **Study area organization**: Data in study_areas/, code in scripts/
3. **Late-fusion architecture**: Keep modality processors separate
4. **No hardcoded paths**: Use configuration files

### Tasks:
1. **Audit imports**:
   - Find any direct `import h3` statements
   - Replace with `from srai.regionalizers import H3Regionalizer`

2. **Fix path handling**:
   - Remove hardcoded paths like 'G:/My Drive/AlphaEarth_Netherlands/'
   - Use configuration files or environment variables

3. **Update documentation**:
   - Ensure README reflects new structure
   - Update ARCHITECTURE.md if needed

## 📝 Implementation Notes

### Order of execution:
1. **FIRST**: Fix security (remove keys, setup secure management)
2. **SECOND**: Commit and push security fixes immediately
3. **THIRD**: Reorganize study_areas structure
4. **FOURTH**: Create new Earth Engine integration
5. **FIFTH**: Test everything works with new structure

### Testing checklist:
- [ ] No exposed credentials in codebase
- [ ] Earth Engine authentication works with .env file
- [ ] All moved scripts still function correctly
- [ ] Import paths updated throughout codebase
- [ ] New Earth Engine script successfully fetches embeddings
- [ ] Study areas only contain data, not scripts

### Git commit strategy:
```bash
# Commit 1: Security fix (URGENT)
git add -A
git commit -m "fix: remove exposed API keys and implement secure credential management"
git push --force origin master

# Commit 2: Reorganization
git add -A  
git commit -m "refactor: reorganize study area scripts to follow architecture guidelines"
git push

# Commit 3: New functionality
git add scripts/earthengine/
git commit -m "feat: add Earth Engine API integration for AlphaEarth embeddings"
git push
```

## 🔍 Additional Context from Exploration

During our exploration with Claude Code, we discovered:
1. No obvious API keys in the main codebase, but user believes one exists in alphaearth scripts
2. The .gitignore is comprehensive but doesn't exclude a keys/ directory
3. Many scripts in study_areas/ that violate the architectural principles
4. Missing proper Earth Engine API integration for fetching embeddings
5. Some scripts still use h3 directly instead of SRAI

## 📞 Contact

If you need clarification on any tasks, the context is documented in:
- CLAUDE.md (architecture principles)  
- ARCHITECTURE.md (system design)
- This conversation with Claude Code (exploration phase)

---

*Generated after exploration session with Claude Code - January 2025*
*These tasks are critical for security and architectural integrity*