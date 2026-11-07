param(
  [string]$Phase = "A",
  [int]$Minutes = 240,
  [int]$EnvCount = 512,
  [int]$Rollout = 192,
  [string]$Output = "outputs/night_league_mpc_A",
  [string]$InitFrom = "outputs/night_run1/checkpoints/hybrid_night_final.pt",
  [string]$FrozenFrom = "outputs/night_run1/checkpoints/initial.pt"
)
$py = "C:\Python314\python.exe"
$repo = Split-Path -Parent $PSScriptRoot
New-Item -ItemType Directory -Force -Path (Join-Path $repo $Output) | Out-Null
$log = Join-Path (Join-Path $repo $Output) "launch.train.log"
$err = Join-Path (Join-Path $repo $Output) "launch.train.err"

if ($Phase -eq "A") {
  $args = @(
    "training/night/train.py", "--minutes", "$Minutes", "--envs", "$EnvCount",
    "--rollout", "$Rollout", "--init-from", $InitFrom, "--frozen-from", $FrozenFrom,
    "--mpc-share", "0.15", "--laika-share", "0.15", "--frozen-share", "0.30",
    "--self-share", "0.40", "--group-advnorm",
    "--lr", "6e-5", "--kl-coef", "0.02", "--entropy-coef", "0.001",
    "--eval-minutes", "30", "--eval-laika-games", "100", "--eval-mpc-games", "50",
    "--eval-h2h-games", "100", "--save-every", "20", "--threads", "12",
    "--output", $Output
  )
} else {
  Write-Error "unknown phase"
  exit 1
}
Start-Process -FilePath $py -ArgumentList $args -WorkingDirectory $repo `
  -RedirectStandardOutput $log -RedirectStandardError $err -WindowStyle Hidden
Write-Output "launched $Phase -> $Output (log: $log)"
