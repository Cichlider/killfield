param(
  [string[]]$Checkpoints,
  [int]$GamesH2H = 2000,
  [int]$GamesLaika = 100,
  [int]$GamesMpc = 100,
  [string]$Tag = "eval"
)
$py = "C:\Python314\python.exe"
$repo = Get-Location
$idx = 0
foreach ($ckpt in $Checkpoints) {
  $name = Split-Path $ckpt -Leaf
  $dir = Join-Path $repo ("outputs/evals/{0}_{1}" -f $Tag, $idx)
  New-Item -ItemType Directory -Force -Path $dir | Out-Null
  Write-Output "== $ckpt"
  & $py training/night/evaluate.py $ckpt --two-bases --games-laika $GamesLaika --games-mpc $GamesMpc --games-h2h $GamesH2H --output (Join-Path $dir "result.json")
  idx++
}
