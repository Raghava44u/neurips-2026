# Overleaf Upload Guide - Fix Missing Plots

## Problem
Your plots/charts/images are not showing in Overleaf when you recompile.

## Root Cause
The `figs/` folder was not uploaded to your Overleaf project. Your LaTeX paper references figures in this folder, but Overleaf cannot find them without the folder being present.

## Evidence
Your paper file (`MemEIC_NeurIPS2026.tex`) contains:
- Line 79: `\graphicspath{{figs/}}` - tells LaTeX to look in figs/ folder
- 5 figure references:
  - Line 554: `\includegraphics[width=\linewidth]{figs/fig3_adv2k_experiments.png}`
  - Line 642: `\includegraphics[width=0.55\linewidth]{figs/fig5_sensitivity.png}`
  - Line 920: `\includegraphics[width=\linewidth]{figs/fig2_category_breakdown.png}`
  - Line 929: `\includegraphics[width=\linewidth]{figs/fig4_ablation.png}`
  - Line 938: `\includegraphics[width=\linewidth]{figs/fig8_connector_gain.png}`

## All 5 Required PNG Files Exist
Located at: `C:\Users\Dr-Prashantkumar\Downloads\MemEIC\MemEIC\NeurIPS\paper\figs\`

✅ fig2_category_breakdown.png
✅ fig3_adv2k_experiments.png
✅ fig4_ablation.png
✅ fig5_sensitivity.png
✅ fig8_connector_gain.png

## NEW: Failure Case Visualization Images
The paper now includes 6 new visual failure analysis figures:

✅ visual_failure_gallery.png - 6-panel showcase of actual MemEIC failures with images and Q/A
✅ visual_confidence_vs_failure.png - Scatter plot showing overconfidence in failures
✅ visual_failure_categories.png - Category-wise failure breakdown heatmap
✅ visual_failure_modes.png - Infographic of 3 failure mode types
✅ visual_pipeline_failures.png - Pipeline failure chain analysis
✅ visual_image_vs_prediction.png - Image vs model prediction comparison

## Solution - Step by Step

### Step 1: Prepare Files Locally
All files are already in: `NeurIPS\paper\figs\`
No action needed - they're ready.

### Step 2: Upload to Overleaf
1. Open your Overleaf project in web browser
2. Click **Menu** (top left)
3. Click **Upload** or **Import**
4. Choose **Upload Folder** option
5. Navigate to: `C:\Users\Dr-Prashantkumar\Downloads\MemEIC\MemEIC\NeurIPS\paper\figs`
6. Select ALL files in the figs folder
7. Upload them

### Step 3: Verify Folder Structure in Overleaf
After upload, your Overleaf project should look like:
```
Overleaf Project/
├── MemEIC_NeurIPS2026.tex
├── figs/
│   ├── fig2_category_breakdown.png
│   ├── fig3_adv2k_experiments.png
│   ├── fig4_ablation.png
│   ├── fig5_sensitivity.png
│   └── fig8_connector_gain.png
└── (other files)
```

### Step 4: Recompile
1. In Overleaf, click **Recompile**
2. Wait for compilation to finish
3. All 5 figures should now appear in your PDF

## What Each Figure Shows
- **fig2_category_breakdown.png** - Performance across 5 failure categories (Polysemy, Near-Miss, Conflict, Multi-hop, Hard Visual)
- **fig3_adv2k_experiments.png** - Comparison of 4 experiments (Exp1-4) with results
- **fig4_ablation.png** - Component importance: shows what happens without Memory, Gating, or LoRA
- **fig5_sensitivity.png** - Hyperparameter tuning: alpha and top-k sensitivity
- **fig8_connector_gain.png** - Gate connector weights for each failure mode

## If Still Not Showing
1. Check that files are in `figs/` subfolder (not just uploaded to root)
2. Recompile again (sometimes needs 2x)
3. Check Overleaf compilation log for file not found errors
4. Make sure filenames match exactly (case-sensitive on some systems)

## Alternative: Upload from Web
If folder upload doesn't work:
1. In Overleaf, create a new folder named `figs`
2. Upload each PNG file individually to that folder
3. Recompile

## Questions?
All PNG files are verified present and ready for upload.
