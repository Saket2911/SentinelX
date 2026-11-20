

$folders = @(
    "data\raw",
    "data\interim",
    "data\processed",
    "src\sentinelx\data",
    "src\sentinelx\features",
    "src\sentinelx\models",
    "src\sentinelx\training",
    "src\sentinelx\evaluation",
    "src\sentinelx\inference",
    "src\sentinelx\explainability",
    "src\sentinelx\utils",
    "scripts",
    "artifacts\models",
    "artifacts\preprocessors",
    "app",
    "tests",
    "reports",
    "logs"
)

foreach ($folder in $folders) {
    New-Item -ItemType Directory -Force -Path $folder | Out-Null
}

New-Item -ItemType File -Force -Path "src\sentinelx\__init__.py" | Out-Null
New-Item -ItemType File -Force -Path "src\sentinelx\data\__init__.py" | Out-Null
New-Item -ItemType File -Force -Path "src\sentinelx\features\__init__.py" | Out-Null
New-Item -ItemType File -Force -Path "src\sentinelx\models\__init__.py" | Out-Null
New-Item -ItemType File -Force -Path "src\sentinelx\training\__init__.py" | Out-Null
New-Item -ItemType File -Force -Path "src\sentinelx\evaluation\__init__.py" | Out-Null
New-Item -ItemType File -Force -Path "src\sentinelx\inference\__init__.py" | Out-Null
New-Item -ItemType File -Force -Path "src\sentinelx\explainability\__init__.py" | Out-Null
New-Item -ItemType File -Force -Path "src\sentinelx\utils\__init__.py" | Out-Null

Write-Host "SentinelX folder structure created."