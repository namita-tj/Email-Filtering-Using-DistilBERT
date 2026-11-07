# Deploying the API on Azure

This puts the spam filter on Azure Container Apps with a public URL, the model in Blob Storage,
secrets in Key Vault and monitoring in Application Insights. After the one-time setup below, every
merge to `main` deploys itself through GitHub Actions.

Everything is defined in [`infra/main.bicep`](../infra/main.bicep). The commands are for Windows
PowerShell, run from the repository folder.

## What it costs

With the default settings (1 vCPU / 2 GiB, scale to zero, at most one replica), the app only uses
compute while it answers requests. Container Apps includes a monthly free grant of compute and
requests, and a portfolio demo should stay inside it; check the current numbers on the Azure
Container Apps pricing page. Log ingestion is capped at 0.2 GB a day, and Bicep sets up a monthly
budget with email alerts at 50% (actual) and 100% (forecast). Everything lives in one resource
group, so `az group delete --name rg-email-filter` removes all of it.

The trade-off of scaling to zero: the first request after a quiet period takes about 1-2 minutes,
because the container starts, downloads the model and loads TensorFlow.

## 0. Once: tools and account

1. An Azure subscription. With a student email, *Azure for Students* gives credit without a credit card;
   otherwise a normal free account.
2. Azure CLI: `winget install -e --id Microsoft.AzureCLI`, then open a new PowerShell window.
3. Sign in and register the providers this uses (only needed once per subscription):

```powershell
az login
az provider register --namespace Microsoft.App
az provider register --namespace Microsoft.OperationalInsights
az provider register --namespace Microsoft.KeyVault
az bicep install
```

## 1. Build the image and make it public

Merge this branch into `main`. CI tests the code and pushes the image to GitHub Container Registry.
The deploy job is skipped until step 5.

Then make the image public, so Azure can pull it without a password: GitHub, your profile,
**Packages**, `email-filtering-using-distilbert`, **Package settings**, **Change visibility**, **Public**.

## 2. Get the trained model

The model is not in the repository (about 260 MB). Train it on Colab with
`notebooks/train_and_evaluate.ipynb` (steps 1 to 5, about 30 minutes), then run this in a new cell
and download the zip:

```python
!cd /content/Email-Filtering-Using-DistilBERT && zip -q -r /content/spam_detection_model.zip spam_detection_model
from google.colab import files
files.download("/content/spam_detection_model.zip")
```

Unzip it into the repository folder, so that `spam_detection_model\tf_model.h5` exists. The folder
is gitignored.

## 3. Create everything except the app

```powershell
az group create --name rg-email-filter --location germanywestcentral
$me = az ad signed-in-user show --query id -o tsv
az deployment group create --resource-group rg-email-filter --template-file infra/main.bicep `
  --parameters deployApp=false adminPrincipalId=$me budgetEmail=you@example.com
```

Use your own email for the budget alerts. If Azure refuses the region (student subscriptions only
allow some), use `westeurope` or `swedencentral` instead. If it fails on the budget, add
`createBudget=false`; the student credit then acts as the spending limit.

## 4. Upload the model

```powershell
$st = az deployment group show --resource-group rg-email-filter --name main `
  --query properties.outputs.storageAccountName.value -o tsv
az storage blob upload-batch --account-name $st --auth-mode login `
  --destination models --destination-path spam_detection_model --source .\spam_detection_model
```

If it says you are not authorised, wait 2-5 minutes: the upload permission from step 3 takes a
moment to apply. The storage account accepts no keys, only signed-in identities.

## 5. Start the app

Same command as step 3, with `deployApp=true`:

```powershell
az deployment group create --resource-group rg-email-filter --template-file infra/main.bicep `
  --parameters deployApp=true adminPrincipalId=$me budgetEmail=you@example.com
az deployment group show --resource-group rg-email-filter --name main --query properties.outputs -o json
```

The outputs include `apiUrl`. Open `<apiUrl>/docs` (give the first start a couple of minutes), or:

```powershell
Invoke-RestMethod -Method Post -Uri "<apiUrl>/predict" -ContentType "application/json" `
  -Body '{"subject": "Account locked", "text": "Verify your password within 24 hours."}'
```

## 6. Connect GitHub for automatic deploys

In the repository on GitHub: **Settings**, **Secrets and variables**, **Actions**, tab **Variables**,
and add these five *variables* (not secrets; they are IDs, not passwords):

| Name | Value |
| --- | --- |
| `AZURE_CLIENT_ID` | the `githubClientId` output |
| `AZURE_TENANT_ID` | the `tenantId` output |
| `AZURE_SUBSCRIPTION_ID` | the `subscriptionId` output |
| `AZURE_RESOURCE_GROUP` | `rg-email-filter` |
| `AZURE_CONTAINERAPP` | `ca-email-filter` |

From now on, every push to `main` builds the image, deploys it and runs a smoke test against the
live URL. To deploy without a new commit: **Actions**, **CI**, **Run workflow**. GitHub signs in
through OpenID Connect as the deploy identity from `main.bicep`, which can only change this
resource group and only accepts tokens from this repository's `main` branch.

## 7. Monitoring

Azure portal, `appi-email-filter`, **Logs**:

```kusto
// requests and latency per hour
requests
| where name has "predict"
| summarize calls = count(), p50_ms = percentile(duration, 50), p95_ms = percentile(duration, 95),
            failed = countif(success == false) by bin(timestamp, 1h)

// what the model decides, and how often the LLM or the rules decide instead
traces
| where message startswith "label="
| summarize count() by label = tostring(customDimensions.label), source = tostring(customDimensions.source)
```

**Live metrics** in the same resource shows requests as they come in. Container logs (startup, model
download) are in `log-email-filter`, table `ContainerAppConsoleLogs_CL`.

## Optional: LLM fallback

Without a key the fallback is off and the model decides alone. With OpenAI credit, add
`openAiApiKey=sk-...` to the step 5 command. The key goes into Key Vault, and the app reads it
through its managed identity.

## Removing it

```powershell
az group delete --name rg-email-filter --yes
```

Key Vault keeps deleted vaults for 7 days. To redeploy within that week under the same name, purge
the old vault first: `az keyvault purge --name <vault name>`.
