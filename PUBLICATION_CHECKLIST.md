# Upload and obtain the software DOI

The source repository is https://github.com/Simhaatt/memristor-no-current. A Zenodo software DOI is obtained by connecting the repository and publishing a GitHub release.

1. Create `memristor-no-current` on GitHub, choose **Public**, and leave the initialize README/license/gitignore options unchecked because they are included here.
2. Confirm the title, creator names and departmental affiliations in `.zenodo.json` and `CITATION.cff` before publishing a release. Zenodo uses `.zenodo.json` for archiving when both files are present; GitHub uses `CITATION.cff` for its citation button. The paper DOI is an article reference, not the software DOI.
3. Use GitHub Desktop: **File > Add local repository**, select this extracted folder, and use **create a repository here** when prompted. Commit the supplied files, then publish it with the public repository destination. Alternatively use the commands below for an empty remote.
4. Open the repository online and confirm the README, `data/splits`, `results`, code and checkpoint inventory are present. Do not upload just the ZIP into the repository as its sole content.
5. Sign into Zenodo, connect GitHub under **Linked accounts**, then open the **GitHub** page, choose **Sync now**, and enable this repository.
6. On GitHub open **Releases > Draft a new release**. Create tag `v1.0.0`, title `v1.0.0 - publication artifacts`, describe the files, and publish it. Enabling Zenodo must precede this release.
7. Return to Zenodo's GitHub page, wait for archival, open the record, and verify that files, creators and software version are correct. Copy the version-specific software DOI.
8. Use that DOI in the proof availability statement below. Reopen its DOI link in a signed-out browser to check access before submitting the proof.

PowerShell/Git commands for the prepared folder and an empty remote:

```powershell
git init -b main
git add .
git commit -m "Release publication code, checkpoints, splits and result tables"
git remote add origin https://github.com/YOUR-OWNER/memristor-no-current.git
git push -u origin main
```

Do not paste a GitHub/Zenodo password or token into a script or the repository. GitHub Desktop handles sign-in for publishing.

## Availability text supported by this snapshot

> Code, archived split definitions, run logs, available trained checkpoints, and numerical result tables are available at https://doi.org/SOFTWARE-DOI (source code: https://github.com/YOUR-OWNER/memristor-no-current). Original datasets are obtained from their cited providers; processed partitions can be regenerated using the supplied preprocessing scripts. The archive documents the regeneration requirements and any unavailable historical checkpoints.

Replace `SOFTWARE-DOI` and `YOUR-OWNER` only with the actual links after release. This wording reflects what is present. The stronger promise of every trained checkpoint and direct access to processed numeric partitions requires filling the documented artifact gaps and clearing data redistribution permissions first.

Official guidance:

- https://help.zenodo.org/docs/profile/linking-accounts/
- https://help.zenodo.org/docs/github/enable-repository/
- https://help.zenodo.org/docs/github/archive-software/github-upload/
- https://docs.github.com/en/repositories/working-with-files/managing-files/adding-a-file-to-a-repository
