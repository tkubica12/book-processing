# Private HTML serving pattern on Azure

Use this pattern when a project produces static HTML/media artifacts that must be internet reachable but private.

## Best architecture

- Azure Blob Storage stores generated files.
- Blob container is private.
- Public blob access is disabled.
- Shared-key access is disabled when tenant policy requires it.
- Azure Container Apps exposes the public HTTPS endpoint.
- The FastAPI app handles GitHub OAuth login.
- Container App has a system-assigned managed identity.
- Managed identity gets `Storage Blob Data Reader` on the storage account.
- App code reads blobs with Azure SDK + `DefaultAzureCredential`.
- Browser never receives blob keys, SAS URLs, or direct blob URLs.

Request flow:

1. User opens site URL.
2. FastAPI redirects unauthenticated users to GitHub OAuth.
3. GitHub redirects back to `/oauth/github/callback`.
4. App exchanges the code for a GitHub token and calls GitHub `/user` and `/user/emails`.
5. App allows only the configured GitHub login and verified email.
6. App issues a signed, HTTP-only, Secure local session cookie.
7. App maps URL path to blob name.
8. App streams blob from private Blob Storage using managed identity.
9. Browser receives HTML/audio/video through Container Apps.

## Why not Azure Storage static website

Storage static website is cheap and simple, but it is anonymous-only. It does not support AuthN/AuthZ on the `$web` endpoint. Do not use it for private content.

## Why not Azure Static Web Apps for large media

Static Web Apps has great auth, but storage limits are too small for large generated media libraries. Use it for small sites, not 100+ GB of audio/video.

## Why not Azure Files mount in Container Apps

Azure Container Apps can mount Azure Files, but the CLI-supported Azure Files mount uses storage account keys. If the tenant disables shared-key access, this fails. Blob + managed identity avoids that.

## Minimal app behavior

The web app should:

- Serve `/` as `index.html`.
- Serve `/folder/` as `folder/index.html`.
- Reject path traversal.
- Set `Cache-Control: private`.
- Support `Range` headers for audio/video seeking.
- Return `206 Partial Content` for ranged media requests.
- Use `DefaultAzureCredential`.
- Implement app-level GitHub OAuth:
  - `/login` starts OAuth.
  - `/oauth/github/callback` validates state, exchanges code, verifies GitHub login/email, and issues a signed session cookie.
  - `/logout` clears local cookies.
  - all blob-serving routes require a valid signed session cookie.
  - only the configured GitHub login and verified email are allowed.

Python packages:

- `fastapi`
- `uvicorn`
- `azure-identity`
- `azure-storage-blob`
- `httpx`

## GitHub OAuth app setup

Create a GitHub OAuth App manually in GitHub Developer settings. GitHub does not expose OAuth App creation through the public REST API or `gh api`, so this step cannot be fully automated.

For `books.tomasonline.net`:

| Field | Value |
|---|---|
| Application name | `Book Processing Private Site` |
| Homepage URL | `https://books.tomasonline.net` |
| Authorization callback URL | `https://books.tomasonline.net/oauth/github/callback` |

Save the generated client ID and client secret. Treat the client secret as a secret; never commit it.

Generate a random local session signing secret:

```powershell
[Convert]::ToBase64String([System.Security.Cryptography.RandomNumberGenerator]::GetBytes(32))
```

The app expects these settings:

- `PUBLIC_BASE_URL`: public origin, for example `https://books.tomasonline.net`
- `GITHUB_OAUTH_CLIENT_ID`: GitHub OAuth App client ID
- `GITHUB_OAUTH_CLIENT_SECRET`: GitHub OAuth App client secret
- `GITHUB_OAUTH_COOKIE_SECRET`: random signing secret for local session cookies
- `ALLOWED_GITHUB_LOGIN`: allowed GitHub username, for example `tkubica12`
- `ALLOWED_GITHUB_EMAIL`: required verified GitHub email, for example `tkubica12@gmail.com`

Store secrets as Container Apps secrets and reference them from env vars:

```powershell
az containerapp secret set `
  --name <app> `
  --resource-group <rg> `
  --secrets `
    "github-oauth-client-secret=<github-client-secret>" `
    "github-oauth-cookie-secret=<random-cookie-secret>"

az containerapp update `
  --name <app> `
  --resource-group <rg> `
  --set-env-vars `
    "PUBLIC_BASE_URL=https://books.tomasonline.net" `
    "GITHUB_OAUTH_CLIENT_ID=<github-client-id>" `
    "GITHUB_OAUTH_CLIENT_SECRET=secretref:github-oauth-client-secret" `
    "GITHUB_OAUTH_COOKIE_SECRET=secretref:github-oauth-cookie-secret" `
    "ALLOWED_GITHUB_LOGIN=tkubica12" `
    "ALLOWED_GITHUB_EMAIL=tkubica12@gmail.com"
```

Disable Container Apps Easy Auth when the app-level OAuth gate is deployed:

```powershell
az containerapp auth update `
  --name <app> `
  --resource-group <rg> `
  --enabled false `
  --yes
```

Do not disable Easy Auth before the container image includes the app-level OAuth gate, otherwise the site can become public.

## Deployment steps

1. Generate local static files.
2. Create StorageV2 account, Standard LRS.
3. Create private blob container.
4. Create ACR.
5. Build web server image.
6. Create Container Apps environment on Consumption profile.
7. Create Container App with system identity.
8. Assign roles:
   - `AcrPull` on ACR
   - `Storage Blob Data Reader` on storage account
   - current deploy user: `Storage Blob Data Contributor` for upload
9. Configure Container App registry pull with system identity.
10. Configure app env vars:
    - `STORAGE_ACCOUNT_NAME`
    - `BLOB_CONTAINER_NAME`
11. Configure ingress to app port.
12. Configure GitHub OAuth env vars and Container Apps secrets.
13. Disable Container Apps Easy Auth after the app-level OAuth gate is deployed.
14. Upload files with AzCopy using Entra login:

```powershell
azcopy login --tenant-id <tenant-id>
azcopy sync output https://<storage>.blob.core.windows.net/<container> --recursive=true --delete-destination=true
```

## Custom domain

For `books.example.com`:

1. Add CNAME:
   - `books` -> Container Apps generated FQDN
2. Add TXT:
   - `asuid.books` -> Container Apps custom domain verification ID
3. Add hostname to Container App.
4. Bind hostname with managed certificate.

Container Apps managed certificate issuance can take several minutes.

## Cost notes

For infrequent access, storage tier matters more than compute.

- Hot: higher storage price, cheapest reads/retrieval.
- Cool: usually best for personal media library with occasional access.
- Cold: cheapest storage, but retrieval and read operations cost more and minimum retention is longer.
- Archive: not suitable for web serving because content must be rehydrated before access.

For a private library where less than 5% of large files are read monthly, prefer Cool. Consider Cold only for content that is rarely opened and can tolerate higher retrieval costs and early-delete constraints.

## Operational tips

- Keep deploy script idempotent.
- Generate indexes before sync.
- Use `--delete-destination=true` only when the local output directory is authoritative.
- Set Container Apps min replicas to `1` during first image/role propagation smoke test, then back to `0`.
- Test unauthenticated access returns `302 Found` to `https://github.com/login/oauth/authorize?...`.
- Test that a non-allowed GitHub account gets `403`.
- Keep Container Apps Easy Auth disabled only when the app-level OAuth gate is present in the running image.
- If a deploy script recreates the app, make sure it sets GitHub OAuth env vars/secrets before disabling Easy Auth.
- Test media with a range request:

```powershell
curl.exe -I -H "Range: bytes=0-99" https://<site>/<file>.mp3
```

Good result: `206 Partial Content`.
