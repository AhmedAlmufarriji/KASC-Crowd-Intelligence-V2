# Publishing to a new GitHub repository

**Intended repository name:** `KASC-Crowd-Intelligence-V2`  
**Account:** `AhmedAlmufarriji`

This project should be published to its own GitHub repository. **Do not overwrite or force-push the original team repository** `BandarAlsarwani/KASC-Crowd-Management`.

## Create the new repository

1. Sign in to GitHub and visit https://github.com/new.
2. Set the owner to `AhmedAlmufarriji` and repository name to `KASC-Crowd-Intelligence-V2`.
3. Description: `Independent synthetic-data stadium crowd forecasting, event-isolated ML evaluation, bilingual dashboard, and Dijkstra routing.`
4. Choose **Public**, or **Private** if you prefer to review the project before making it visible.
5. Leave **Add a README**, **Add .gitignore**, and **Choose a license** unset. They're already handled by this package or intentionally left undecided.
6. Select **Create repository**.

## Upload from the downloaded ZIP (Windows PowerShell)

Extract the ZIP. Open PowerShell inside the directory that contains `README.md`, `pyproject.toml`, `src/`, `static/`, etc. If Git is installed and your GitHub account is authenticated:

```powershell
git init
git branch -M main
git add .
git commit -m "Initial KASC Crowd Intelligence V2 research prototype"
git remote add origin https://github.com/AhmedAlmufarriji/KASC-Crowd-Intelligence-V2.git
git push -u origin main
```

If Git prompts for a password over HTTPS, use Git Credential Manager or the browser-based sign-in flow rather than pasting your GitHub account password. Do not share access tokens in chat.

## Verify the publication

- The GitHub repository root should show the bilingual dashboard screenshot and project overview.
- The `main` branch should contain `src/`, `static/`, `tests/`, `docs/`, `data/`, `reports/`, `.github/workflows/ci.yml`.
- The GitHub Actions **V2 checks** workflow should pass.
- Confirm no private source materials from the original university project were added.
- The project is not automatically hosted as a live app by GitHub; the FastAPI backend must run locally or on a separately configured hosting environment.

**GitHub connection note:** Existing-repository content editing is available to the ChatGPT GitHub integration in this session, but creating new repositories was not. Once an empty new repo exists, the files can also be uploaded using a capable GitHub connection rather than a manual push.
