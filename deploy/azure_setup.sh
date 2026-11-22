#!/usr/bin/env bash
# One-time Azure setup for the public RideSafe AI demo. Run it in Azure Cloud Shell (Bash):
#
#   git clone https://github.com/Md-Faiz-Alam/helmet-violation-detection.git
#   bash helmet-violation-detection/deploy/azure_setup.sh
#
# It creates, all inside one resource group (delete the group to remove everything):
#   - a Container Apps environment + the API container app (image from ghcr.io; scales to zero, at most 1 replica)
#   - a Static Web App, Free tier (the React app)
#   - an Entra app registration that GitHub Actions signs in with through OIDC (no stored password), allowed to
#     change this resource group only
#   - a monthly budget alert on the subscription
# and prints the GitHub repository variables the deploy workflow needs. Safe to run again: existing resources are
# kept, and the author key is only generated the first time (ROTATE_KEYS=1 makes new ones).
#
# Cost notes: no Log Analytics workspace (logs: `az containerapp logs show`), no container registry (GHCR is free
# for public images), no always-on replica. Idle costs nothing; an active replica (1 vCPU, 2 GiB) uses the
# monthly Container Apps free grant first, which is shared with your other container apps.
#
# Settings (environment variables), defaults in brackets:
#   GH_REPO [Md-Faiz-Alam/helmet-violation-detection]   LOCATION [centralindia]   SWA_LOCATION [first allowed of
#   eastasia, westeurope, eastus2, centralus, westus2]   RG [ridesafe-rg]   BUDGET_USD [10]   ROTATE_KEYS [0]

set -euo pipefail

GH_REPO="${GH_REPO:-Md-Faiz-Alam/helmet-violation-detection}"
LOCATION="${LOCATION:-centralindia}"
RG="${RG:-ridesafe-rg}"
ENV_NAME="${ENV_NAME:-ridesafe-env}"
APP="${APP:-ridesafe-api}"
SWA="${SWA:-ridesafe-web}"
APP_REG="${APP_REG:-ridesafe-github-deploy}"
BUDGET_USD="${BUDGET_USD:-10}"
ROTATE_KEYS="${ROTATE_KEYS:-0}"
IMAGE="ghcr.io/${GH_REPO,,}/ridesafe-api"
TAGS=(project=ridesafe)

step() { printf '\n\033[1;33m==> %s\033[0m\n' "$*"; }
info() { printf '    %s\n' "$*"; }
fail() { printf '\n\033[1;31mERROR: %s\033[0m\n' "$*" >&2; exit 1; }

# ------------------------------------------------------------------------------------------------ preflight
step "Subscription"
az account show >/dev/null 2>&1 || fail "Not signed in. In Cloud Shell this is automatic; elsewhere run: az login"
SUB_ID=$(az account show --query id -o tsv)
TENANT_ID=$(az account show --query tenantId -o tsv)
SUB_NAME=$(az account show --query name -o tsv)
info "Using: $SUB_NAME ($SUB_ID)"
info "If that is the wrong subscription: az account set --subscription <name or id>, then run this again."

step "Allowed regions"
# Azure for Students subscriptions restrict regions with an "allowed locations" policy
ALLOWED=$(az policy assignment list --disable-scope-strict-match \
  --query "[].parameters.listOfAllowedLocations.value[]" -o tsv 2>/dev/null | tr 'A-Z' 'a-z' | sort -u || true)
allowed() { [ -z "$ALLOWED" ] || grep -qx "$1" <<<"$ALLOWED"; }
if [ -n "$ALLOWED" ]; then info "Your subscription allows: $(tr '\n' ' ' <<<"$ALLOWED")"; else info "No region policy found"; fi
allowed "$LOCATION" || fail "$LOCATION is not allowed. Pick one of the regions above, e.g.: LOCATION=<region> bash $0"
if [ -z "${SWA_LOCATION:-}" ]; then
  # Static Web Apps exists in a few regions only; the site itself is served from a global CDN either way
  for r in eastasia westeurope eastus2 centralus westus2; do allowed "$r" && SWA_LOCATION=$r && break; done
fi
[ -n "${SWA_LOCATION:-}" ] || fail "None of the Static Web Apps regions is allowed on this subscription"
info "API region: $LOCATION, web app region: $SWA_LOCATION"

step "Container image"
# The CI workflow pushes the image on every change to main; Azure pulls it anonymously, so it must be public
TOKEN=$(curl -fsS "https://ghcr.io/token?scope=repository:${IMAGE#ghcr.io/}:pull" | sed -E 's/.*"token":"([^"]+)".*/\1/' || true)
if ! curl -fsS -o /dev/null -H "Authorization: Bearer $TOKEN" \
     -H "Accept: application/vnd.oci.image.index.v1+json, application/vnd.docker.distribution.manifest.v2+json" \
     "https://ghcr.io/v2/${IMAGE#ghcr.io/}/manifests/latest"; then
  fail "Cannot pull $IMAGE:latest anonymously. Push to main first (CI publishes the image), then on GitHub open
       Packages > ridesafe-api > Package settings > Change visibility > Public, and run this again."
fi
info "$IMAGE:latest is public"

step "Resource providers"
for ns in Microsoft.App Microsoft.Web; do
  az provider register --namespace "$ns" --wait >/dev/null
  info "$ns registered"
done
az extension add --name containerapp --upgrade --only-show-errors -y >/dev/null

# ------------------------------------------------------------------------------------------------ resources
step "Resource group $RG"
az group create -n "$RG" -l "$LOCATION" --tags "${TAGS[@]}" -o none
info "ok"

step "Static Web App $SWA (Free)"
if ! az staticwebapp show -n "$SWA" -g "$RG" -o none 2>/dev/null; then
  az staticwebapp create -n "$SWA" -g "$RG" -l "$SWA_LOCATION" --sku Free --tags "${TAGS[@]}" -o none
fi
WEB_HOST=$(az staticwebapp show -n "$SWA" -g "$RG" --query defaultHostname -o tsv)
WEB_URL="https://$WEB_HOST"
# Pull request previews live on <name>-<PR number>.<region>...azurestaticapps.net
CORS_REGEX="https://${WEB_HOST%%.*}-[0-9]+\.([a-z0-9-]+\.)*azurestaticapps\.net"
info "$WEB_URL"

step "Container Apps environment $ENV_NAME (no Log Analytics: nothing billed while idle)"
if ! az containerapp env show -n "$ENV_NAME" -g "$RG" -o none 2>/dev/null; then
  az containerapp env create -n "$ENV_NAME" -g "$RG" -l "$LOCATION" --logs-destination none \
    --tags "${TAGS[@]}" -o none
fi
info "ok"

step "Container app $APP"
ENV_VARS=(
  RIDESAFE_ADMIN_TOKEN=secretref:admin-token
  RIDESAFE_URL_SECRET=secretref:url-secret
  RIDESAFE_DEMO=true
  RIDESAFE_TRUSTED_PROXY=true
  RIDESAFE_UPLOAD_TTL_S=3600
  "RIDESAFE_CORS_ORIGINS=$WEB_URL"
  "RIDESAFE_CORS_ORIGIN_REGEX=$CORS_REGEX"
)
ADMIN_KEY=""
if ! az containerapp show -n "$APP" -g "$RG" -o none 2>/dev/null; then
  ADMIN_KEY=$(openssl rand -hex 32)
  # One replica at most: video jobs and their state live in the API process
  az containerapp create -n "$APP" -g "$RG" --environment "$ENV_NAME" \
    --image "$IMAGE:latest" --target-port 8000 --ingress external \
    --cpu 1.0 --memory 2.0Gi --min-replicas 0 --max-replicas 1 \
    --secrets "admin-token=$ADMIN_KEY" "url-secret=$(openssl rand -hex 32)" \
    --env-vars "${ENV_VARS[@]}" --tags "${TAGS[@]}" -o none
else
  if [ "$ROTATE_KEYS" = "1" ]; then
    ADMIN_KEY=$(openssl rand -hex 32)
    az containerapp secret set -n "$APP" -g "$RG" -o none \
      --secrets "admin-token=$ADMIN_KEY" "url-secret=$(openssl rand -hex 32)"
  fi
  az containerapp update -n "$APP" -g "$RG" --min-replicas 0 --max-replicas 1 \
    --set-env-vars "${ENV_VARS[@]}" -o none
fi
API_URL="https://$(az containerapp show -n "$APP" -g "$RG" --query properties.configuration.ingress.fqdn -o tsv)"
info "$API_URL"

# ------------------------------------------------------------------------------------------------ GitHub OIDC
step "GitHub Actions sign-in ($APP_REG, OIDC)"
CLIENT_ID=$(az ad app list --display-name "$APP_REG" --query "[0].appId" -o tsv)
if [ -z "$CLIENT_ID" ]; then
  CLIENT_ID=$(az ad app create --display-name "$APP_REG" --query appId -o tsv)
fi
SP_ID=$(az ad sp list --filter "appId eq '$CLIENT_ID'" --query "[0].id" -o tsv)
if [ -z "$SP_ID" ]; then
  SP_ID=$(az ad sp create --id "$CLIENT_ID" --query id -o tsv)
fi
add_credential() {  # name, subject
  if [ -z "$(az ad app federated-credential list --id "$CLIENT_ID" --query "[?name=='$1'].name" -o tsv)" ]; then
    az ad app federated-credential create --id "$CLIENT_ID" -o none --parameters \
      "{\"name\":\"$1\",\"issuer\":\"https://token.actions.githubusercontent.com\",\"subject\":\"$2\",\"audiences\":[\"api://AzureADTokenExchange\"]}"
  fi
}
add_credential github-main "repo:$GH_REPO:ref:refs/heads/main"          # deploys from main
add_credential github-pull-requests "repo:$GH_REPO:pull_request"        # web app previews for pull requests
RG_SCOPE="/subscriptions/$SUB_ID/resourceGroups/$RG"
if [ -z "$(az role assignment list --assignee "$SP_ID" --scope "$RG_SCOPE" --role Contributor --query "[0].id" -o tsv)" ]; then
  # A new service principal can take a few seconds to become visible to role assignments
  for i in 1 2 3 4 5 6; do
    az role assignment create --assignee-object-id "$SP_ID" --assignee-principal-type ServicePrincipal \
      --role Contributor --scope "$RG_SCOPE" -o none 2>/dev/null && break
    sleep 10
  done
fi
info "Contributor on $RG only"

# ------------------------------------------------------------------------------------------------ budget
step "Budget alert: \$$BUDGET_USD per month on the subscription (50%, 80%, 100% and forecast 100%)"
EMAIL=$(az account show --query user.name -o tsv)
START=$(date -u +%Y-%m-01T00:00:00Z)
BUDGET_BODY=$(cat <<JSON
{"properties": {"category": "Cost", "amount": $BUDGET_USD, "timeGrain": "Monthly",
  "timePeriod": {"startDate": "$START"},
  "notifications": {
    "actual50":    {"enabled": true, "operator": "GreaterThan", "threshold": 50,  "contactEmails": ["$EMAIL"]},
    "actual80":    {"enabled": true, "operator": "GreaterThan", "threshold": 80,  "contactEmails": ["$EMAIL"]},
    "actual100":   {"enabled": true, "operator": "GreaterThan", "threshold": 100, "contactEmails": ["$EMAIL"]},
    "forecast100": {"enabled": true, "operator": "GreaterThan", "threshold": 100, "contactEmails": ["$EMAIL"],
                    "thresholdType": "Forecasted"}}}}
JSON
)
if az rest --method put -o none --body "$BUDGET_BODY" \
     --url "https://management.azure.com/subscriptions/$SUB_ID/providers/Microsoft.Consumption/budgets/ridesafe-monthly?api-version=2023-05-01" 2>/tmp/budget.err; then
  info "Alerts go to $EMAIL"
else
  info "Could not create the budget here ($(head -c 300 /tmp/budget.err))."
  info "Create it in the portal instead: Cost Management > Budgets > Add, \$$BUDGET_USD monthly, alerts at 50/80/100%."
fi

# ------------------------------------------------------------------------------------------------ summary
step "Done"
cat <<EOF

  API:      $API_URL   (docs: $API_URL/docs; the first request after idle takes ~30 s while it starts)
  Web app:  $WEB_URL   (empty until the first deploy)

Next, on your PC in PowerShell (gh is signed in there), set the repository variables the deploy workflow reads.
These are IDs, not passwords; GitHub signs in to Azure with OIDC:

  gh variable set AZURE_CLIENT_ID --repo $GH_REPO --body $CLIENT_ID
  gh variable set AZURE_TENANT_ID --repo $GH_REPO --body $TENANT_ID
  gh variable set AZURE_SUBSCRIPTION_ID --repo $GH_REPO --body $SUB_ID
  gh variable set RIDESAFE_API_URL --repo $GH_REPO --body $API_URL

Then deploy everything once (later pushes to main deploy automatically):

  gh workflow run ci.yml --repo $GH_REPO --ref main
EOF
if [ -n "$ADMIN_KEY" ]; then
  cat <<EOF

Your author key for the live site (shown only now; save it in a password manager, never in git).
Sign in with it from the web app's sidebar, "Author sign-in":

  $ADMIN_KEY
EOF
fi
