# Football predictor (auto-updating)

Every day a free GitHub job downloads the latest Premier League results and fixtures from
football-data.co.uk, refits the models and republishes the page. The page changes after each
game week, and rolls over to a new season by itself (new clubs start at league-average ratings).
If a download fails, nothing is published, so the last good page stays online.

## Set-up (about 10 minutes, works from a phone browser)
1. Make a free account at github.com and create a **public** repository called `football-predictor`.
2. In the repository tap **Add file > Upload files** and upload: `build_app.py`, `template.html`,
   `requirements.txt`, `README.md` and the `site` folder (or just `site/index.html`).
3. Tap **Add file > Create new file**. In the name box type `.github/workflows/update.yml`
   (typing the slashes makes the folders), paste in the contents of `update.yml`, then **Commit**.
4. Go to **Settings > Pages**, and under **Source** choose **GitHub Actions**.
5. Go to the **Actions** tab, choose **Update predictions**, tap **Run workflow**. After a minute or two the
   address shown under Settings > Pages (like `https://YOURNAME.github.io/football-predictor/`) is your live app.

From then on it updates itself every day. Open the page and the heading shows the matchweek and the date of the latest results.

## Good to know
- Results appear on football-data.co.uk a day or so after matches, so predictions for the next game week
  are usually refreshed by Monday or Tuesday.
- Only the Premier League updates automatically. The other leagues in the app are still fixed snapshots.
- GitHub can pause scheduled jobs on public repos with no activity for 60 days. The job records a small daily
  note to avoid this, but if the Actions tab ever says the schedule is disabled, tap **Enable workflow**.
- Already using Netlify? The simplest route is to use the GitHub Pages address from step 5 instead. Keeping Netlify needs one extra
  step (a build hook), so ask for it if you want that.
- To test on a computer: `pip install -r requirements.txt` then `python build_app.py --out site/index.html`.
