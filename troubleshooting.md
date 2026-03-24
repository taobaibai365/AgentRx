# Troubleshooting

## Azure Authentication

### `DefaultAzureCredential` fails on first run (IMDS timeout)

**Symptom:** The first pipeline run fails with a long error listing all attempted credentials:
```
DefaultAzureCredential failed to retrieve a token from the included credentials.
Attempted credentials:
  ManagedIdentityCredential: ManagedIdentityCredential authentication unavailable, no response from the IMDS endpoint.
  AzureCliCredential: Failed to invoke the Azure CLI
  ...
```

**Cause:** `DefaultAzureCredential` tries `ManagedIdentityCredential` early in its chain. On a local dev machine, this attempts to contact the IMDS endpoint which doesn't exist locally, so it blocks until the network timeout (~5-10s). This can exhaust the overall credential chain timeout or cause cascading failures before `AzureCliCredential` gets a chance to run.

This is a known issue across all Azure SDKs: [azure-sdk-for-python #35452](https://github.com/Azure/azure-sdk-for-python/issues/35452)

**Workarounds:**

1. **Retry** — subsequent runs typically succeed because the token gets cached.

2. **Warm the cache first** — run this before the pipeline:
   ```bash
   .venv/Scripts/python -c "from azure.identity import AzureCliCredential; AzureCliCredential().get_token('https://cognitiveservices.azure.com/.default'); print('Token cached')"
   ```

3. **Exclude ManagedIdentityCredential** — in `src/llm_clients/azure.py`, change:
   ```python
   DefaultAzureCredential(managed_identity_client_id=g.CLIENT_ID)
   ```
   to:
   ```python
   DefaultAzureCredential(exclude_managed_identity_credential=True)
   ```
   This skips the IMDS probe entirely. Only use this for local dev — do not commit if running in Azure (where managed identity is needed).

### Prerequisite: `az login`

The pipeline uses Azure AD token-based auth. You must be logged in:
```bash
az login
az account show  # verify
```
