# Market Sentiment Dashboard — PDF-style web edition

This version preserves the **layout and graphs** of the original PDF while making the indicator names clickable and charts interactive. It does **not** recalculate on every page view. GitHub Actions runs Python on weekdays at 6:17 p.m. `America/Toronto`, then publishes the updated static website on GitHub Pages.

## Local setup (Windows / PowerShell)

From this folder:

```powershell
py -m pip install -r requirements.txt
py build.py
py -m http.server 8000 --directory site
```

Open http://localhost:8000/ . The first `build.py` run downloads the Yahoo Finance and iShares inputs and may take some time. The dashboard displays the **latest successfully published** snapshot; market holidays or download failures may leave an older market date.

## Publish on GitHub Pages

1. Create a **public** repository (unless you have a plan that supports private Pages). Upload **the contents of this folder**, including `.github/workflows/deploy.yml`.
2. In **Settings → Pages → Build and deployment**, choose **GitHub Actions**.
3. In **Actions**, select **Update and publish dashboard → Run workflow**. The workflow also runs on pushes to `main` and automatically on weekdays at 18:17 Toronto time.
4. Check the run logs if the build fails. The generated website will be at `https://USERNAME.github.io/REPOSITORY/`.

The script keeps the original PDF report and builds a separate JSON snapshot used by the website. It does not commit the latest snapshots back to Git; GitHub Pages receives them as deployment artifacts.

## Main files

- `legacy_report.py`: original PDF/report engine; left unchanged.
- `build.py`: runs the original engine and serializes its calculated outputs.
- `site/index.html`: dashboard HTML pages.
- `site/styles.css`: PDF-like styling.
- `site/app.js`: table routing and interactive Plotly charts.
- `.github/workflows/deploy.yml`: scheduled data refresh and Pages deployment.

**Important:** The website is public. Remove institutional branding unless you have permission, and publish only public data. Yahoo Finance and iShares availability and redistribution terms should be reviewed for public deployment. A build failure does not publish new data. The PDF charts remain authoritative for strict visual comparison: web charts follow the same data and design but may not be pixel-identical.
