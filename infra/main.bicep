// Everything the spam filter API needs on Azure, in one resource group.
//
//   Container Apps    the API, scaled to zero when nobody uses it (about free when idle)
//   Blob Storage      the trained model; the app reads it with its managed identity, no keys
//   Key Vault         secrets (App Insights connection string, optional LLM key), read by the app's identity
//   App Insights      request traces, latency and prediction logs, on a Log Analytics workspace
//   Deploy identity   lets GitHub Actions deploy through OIDC, so no Azure password is stored in GitHub
//   Budget            an email alert before costs get anywhere near real money
//
// Deploy in two steps (see docs/deploy-azure.md): first with deployApp=false, upload the model,
// then again with deployApp=true.

targetScope = 'resourceGroup'

@description('Azure region for everything.')
param location string = resourceGroup().location

@description('Short name used in resource names.')
@maxLength(12)
param appName string = 'email-filter'

@description('Container image, e.g. ghcr.io/namita-tj/email-filtering-using-distilbert:latest')
param image string = 'ghcr.io/namita-tj/email-filtering-using-distilbert:latest'

@description('Create the container app. False on the first run, before the model is uploaded.')
param deployApp bool = true

@description('Your Entra object ID (az ad signed-in-user show --query id -o tsv): lets you upload the model and manage secrets.')
param adminPrincipalId string

@description('GitHub repository allowed to deploy, as owner/name (case as on GitHub).')
param githubRepository string = 'namita-tj/Email-Filtering-Using-DistilBERT'

@description('Branch that deploys.')
param githubBranch string = 'main'

@description('vCPU per replica. Memory is twice this in Gi (Consumption plan rule).')
@allowed(['0.5', '1.0', '1.5', '2.0'])
param cpu string = '1.0'

@description('Most replicas. 1 keeps the cost predictable for a portfolio demo.')
@minValue(1)
@maxValue(5)
param maxReplicas int = 1

@description('Optional OpenAI API key for the LLM fallback. Empty: fallback off.')
@secure()
param openAiApiKey string = ''

@description('Create a monthly cost budget with email alerts. Set false if your subscription type does not support budgets.')
param createBudget bool = true

@description('Monthly budget in the subscription currency.')
param monthlyBudget int = 10

@description('Email for budget alerts.')
param budgetEmail string = ''

@description('Budget start, first day of the current month.')
param budgetStartDate string = utcNow('yyyy-MM-01')

var suffix = uniqueString(resourceGroup().id)
var storageName = 'st${replace(appName, '-', '')}${take(suffix, 6)}'
var vaultName = 'kv-${appName}-${take(suffix, 6)}'
var modelContainer = 'models'
var modelFolder = 'spam_detection_model'
var useLlm = !empty(openAiApiKey)
var memory = { '0.5': '1Gi', '1.0': '2Gi', '1.5': '3Gi', '2.0': '4Gi' }[cpu] // ARM arithmetic is integer-only

var roles = {
  blobReader: '2a2b9908-6ea1-4ae2-8e65-a410df84e7d1' // Storage Blob Data Reader
  blobContributor: 'ba92f5b4-2d11-453d-a403-e96b0029c9fe' // Storage Blob Data Contributor
  secretsUser: '4633458b-17de-408a-b874-0445c86b69e6' // Key Vault Secrets User
  secretsOfficer: 'b86a8fe4-44ce-4948-aee5-eccb2c155cd7' // Key Vault Secrets Officer
  contributor: 'b24988ac-6180-42a0-ab88-20f7382dd24c' // Contributor
}

// ── Monitoring ─────────────────────────────────────────────────────────────

resource logs 'Microsoft.OperationalInsights/workspaces@2023-09-01' = {
  name: 'log-${appName}'
  location: location
  properties: {
    sku: { name: 'PerGB2018' }
    retentionInDays: 30
    workspaceCapping: { dailyQuotaGb: json('0.2') } // caps log ingestion, and so its cost
  }
}

resource insights 'Microsoft.Insights/components@2020-02-02' = {
  name: 'appi-${appName}'
  location: location
  kind: 'web'
  properties: {
    Application_Type: 'web'
    WorkspaceResourceId: logs.id
  }
}

// ── Identities ─────────────────────────────────────────────────────────────

@description('The identity the running app uses to read the model and its secrets.')
resource appIdentity 'Microsoft.ManagedIdentity/userAssignedIdentities@2023-01-31' = {
  name: 'id-${appName}'
  location: location
}

@description('The identity GitHub Actions signs in as (OIDC, no stored password).')
resource deployIdentity 'Microsoft.ManagedIdentity/userAssignedIdentities@2023-01-31' = {
  name: 'id-${appName}-github'
  location: location
}

resource githubCredential 'Microsoft.ManagedIdentity/userAssignedIdentities/federatedIdentityCredentials@2023-01-31' = {
  parent: deployIdentity
  name: 'github-${githubBranch}'
  properties: {
    issuer: 'https://token.actions.githubusercontent.com'
    subject: 'repo:${githubRepository}:ref:refs/heads/${githubBranch}'
    audiences: ['api://AzureADTokenExchange']
  }
}

resource deployCanUpdateApp 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(resourceGroup().id, deployIdentity.id, roles.contributor)
  properties: {
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', roles.contributor)
    principalId: deployIdentity.properties.principalId
    principalType: 'ServicePrincipal'
  }
}

// ── Model storage ──────────────────────────────────────────────────────────

resource storage 'Microsoft.Storage/storageAccounts@2023-05-01' = {
  name: storageName
  location: location
  sku: { name: 'Standard_LRS' }
  kind: 'StorageV2'
  properties: {
    minimumTlsVersion: 'TLS1_2'
    allowBlobPublicAccess: false
    allowSharedKeyAccess: false // identities only, no account keys
    supportsHttpsTrafficOnly: true
  }
}

resource blobService 'Microsoft.Storage/storageAccounts/blobServices@2023-05-01' = {
  parent: storage
  name: 'default'
}

resource models 'Microsoft.Storage/storageAccounts/blobServices/containers@2023-05-01' = {
  parent: blobService
  name: modelContainer
  properties: { publicAccess: 'None' }
}

resource appReadsModel 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(storage.id, appIdentity.id, roles.blobReader)
  scope: storage
  properties: {
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', roles.blobReader)
    principalId: appIdentity.properties.principalId
    principalType: 'ServicePrincipal'
  }
}

resource adminUploadsModel 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(storage.id, adminPrincipalId, roles.blobContributor)
  scope: storage
  properties: {
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', roles.blobContributor)
    principalId: adminPrincipalId
    principalType: 'User'
  }
}

// ── Secrets ────────────────────────────────────────────────────────────────

resource vault 'Microsoft.KeyVault/vaults@2023-07-01' = {
  name: vaultName
  location: location
  properties: {
    tenantId: subscription().tenantId
    sku: { family: 'A', name: 'standard' }
    enableRbacAuthorization: true
    enableSoftDelete: true
    softDeleteRetentionInDays: 7
  }
}

resource insightsSecret 'Microsoft.KeyVault/vaults/secrets@2023-07-01' = {
  parent: vault
  name: 'appinsights-connection-string'
  properties: { value: insights.properties.ConnectionString }
}

resource llmSecret 'Microsoft.KeyVault/vaults/secrets@2023-07-01' = if (useLlm) {
  parent: vault
  name: 'openai-api-key'
  properties: { value: openAiApiKey }
}

resource appReadsSecrets 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(vault.id, appIdentity.id, roles.secretsUser)
  scope: vault
  properties: {
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', roles.secretsUser)
    principalId: appIdentity.properties.principalId
    principalType: 'ServicePrincipal'
  }
}

resource adminManagesSecrets 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(vault.id, adminPrincipalId, roles.secretsOfficer)
  scope: vault
  properties: {
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', roles.secretsOfficer)
    principalId: adminPrincipalId
    principalType: 'User'
  }
}

// ── The API ────────────────────────────────────────────────────────────────

resource environment 'Microsoft.App/managedEnvironments@2024-03-01' = {
  name: 'cae-${appName}'
  location: location
  properties: {
    appLogsConfiguration: {
      destination: 'log-analytics'
      logAnalyticsConfiguration: {
        customerId: logs.properties.customerId
        sharedKey: logs.listKeys().primarySharedKey
      }
    }
  }
}

var baseSecrets = [
  {
    name: 'appinsights-connection-string'
    keyVaultUrl: insightsSecret.properties.secretUri
    identity: appIdentity.id
  }
]
var llmSecrets = useLlm ? [
  {
    name: 'openai-api-key'
    keyVaultUrl: '${vault.properties.vaultUri}secrets/openai-api-key' // built from the vault URI: the secret is conditional
    identity: appIdentity.id
  }
] : []

var baseEnv = [
  { name: 'MODEL_BLOB_URL', value: '${storage.properties.primaryEndpoints.blob}${modelContainer}/${modelFolder}/' }
  { name: 'AZURE_CLIENT_ID', value: appIdentity.properties.clientId } // which identity DefaultAzureCredential uses
  { name: 'APPLICATIONINSIGHTS_CONNECTION_STRING', secretRef: 'appinsights-connection-string' }
  { name: 'LLM_PROVIDER', value: useLlm ? 'openai' : 'none' }
  { name: 'LOG_LEVEL', value: 'INFO' }
]
var llmEnv = useLlm ? [{ name: 'OPENAI_API_KEY', secretRef: 'openai-api-key' }] : []

resource api 'Microsoft.App/containerApps@2024-03-01' = if (deployApp) {
  name: 'ca-${appName}'
  location: location
  identity: {
    type: 'UserAssigned'
    userAssignedIdentities: { '${appIdentity.id}': {} }
  }
  properties: {
    managedEnvironmentId: environment.id
    configuration: {
      activeRevisionsMode: 'Single'
      ingress: {
        external: true
        targetPort: 8000
        transport: 'auto'
        allowInsecure: false
      }
      secrets: concat(baseSecrets, llmSecrets)
    }
    template: {
      containers: [
        {
          name: 'api'
          image: image
          resources: { cpu: json(cpu), memory: memory }
          env: concat(baseEnv, llmEnv)
          probes: [
            {
              // The model downloads and loads before the server listens: allow ~4 minutes.
              type: 'Startup'
              httpGet: { path: '/health', port: 8000 }
              initialDelaySeconds: 30
              periodSeconds: 20
              failureThreshold: 10
            }
            {
              type: 'Liveness'
              httpGet: { path: '/health', port: 8000 }
              periodSeconds: 30
              failureThreshold: 3
            }
          ]
        }
      ]
      scale: {
        minReplicas: 0 // scale to zero: no cost while idle, ~1-2 min cold start on the next request
        maxReplicas: maxReplicas
        rules: [
          {
            name: 'http'
            http: { metadata: { concurrentRequests: '10' } }
          }
        ]
      }
    }
  }
  dependsOn: [appReadsModel, appReadsSecrets, llmSecret]
}

// ── Cost guard ─────────────────────────────────────────────────────────────

resource budget 'Microsoft.Consumption/budgets@2023-11-01' = if (createBudget && !empty(budgetEmail)) {
  name: 'budget-${appName}'
  properties: {
    category: 'Cost'
    amount: monthlyBudget
    timeGrain: 'Monthly'
    timePeriod: { startDate: budgetStartDate }
    notifications: {
      actual50: {
        enabled: true
        operator: 'GreaterThanOrEqualTo'
        threshold: 50
        thresholdType: 'Actual'
        contactEmails: [budgetEmail]
      }
      forecast100: {
        enabled: true
        operator: 'GreaterThanOrEqualTo'
        threshold: 100
        thresholdType: 'Forecasted'
        contactEmails: [budgetEmail]
      }
    }
  }
}

// ── Outputs: what the setup steps and GitHub need ──────────────────────────

output apiUrl string = deployApp ? 'https://${api!.properties.configuration.ingress.fqdn}' : ''
output containerAppName string = 'ca-${appName}'
output storageAccountName string = storage.name
output modelContainer string = modelContainer
output modelFolder string = modelFolder
output keyVaultName string = vault.name
output githubClientId string = deployIdentity.properties.clientId
output tenantId string = subscription().tenantId
output subscriptionId string = subscription().subscriptionId
