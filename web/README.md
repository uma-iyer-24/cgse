# CGSE — results preview (static site)

Small **static** page to show **Tier 1 / Tier 1b** metrics (tables + Chart.js), plus the **concept PNGs** from `paper_documentation/figures/`.

## Build

From the **repository root**:

```bash
python scripts/build_results_site.py
```

This reads `runs/tier1/metrics/*.csv` and `runs/tier1b/metrics/*.csv`, writes `web/data/results.json` and `web/generated-config.js`, and copies figures into `web/assets/figures/`.

## View

**Option A — double-click** `index.html` (works because data is inlined in `generated-config.js`).

**Option B — local server** (useful if you change to `fetch()` later):

```bash
cd web
python -m http.server 8765
```

Then open [http://localhost:8765](http://localhost:8765).

## GitHub Pages (optional)

1. Repository **Settings → Pages → Build and deployment**: set **Source** to **GitHub Actions**.
2. Push to **`main`** or **`phase1-graph`** (or trigger **Actions → Deploy results site** manually).
3. The workflow **`.github/workflows/deploy-results-site.yml`** runs `build_results_site.py` on the runner and publishes **`web/`**. Your site URL will look like `https://<user>.github.io/cgse/` (exact URL appears in the workflow run and Pages settings).

**Note:** The runner only sees **committed** `runs/**` CSVs. Commit new metrics before expecting them on Pages.

## After new training runs

Re-run `build_results_site.py` so charts reflect new CSVs. For long sweeps, run training first, then rebuild.
