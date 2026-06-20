[CmdletBinding()]
param(
    [string]$SourceRoot = (Join-Path -Path $PSScriptRoot -ChildPath '..\output'),
    [string]$DestinationRoot = 'C:\Users\tokubica\OneDrive\books-ai-processed'
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

function Copy-DirectoryTree {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)]
        [string]$SourcePath,

        [Parameter(Mandatory)]
        [string]$DestinationPath
    )

    $result = [pscustomobject]@{
        DirectoriesCreated = 0
        FilesCopied       = 0
        DirectoriesSkipped = 0
    }

    if (-not (Test-Path -LiteralPath $DestinationPath)) {
        $null = New-Item -ItemType Directory -Path $DestinationPath -Force
        $result.DirectoriesCreated++
    }

    foreach ($child in Get-ChildItem -LiteralPath $SourcePath -Force) {
        $targetPath = Join-Path -Path $DestinationPath -ChildPath $child.Name

        if ($child.PSIsContainer) {
            if ($child.Name -like '*_partial*') {
                Write-Host ("    [SKIP] Nested partial folder: {0}" -f $child.FullName) -ForegroundColor Yellow
                $result.DirectoriesSkipped++
                continue
            }

            $childResult = Copy-DirectoryTree -SourcePath $child.FullName -DestinationPath $targetPath
            $result.DirectoriesCreated += $childResult.DirectoriesCreated
            $result.FilesCopied += $childResult.FilesCopied
            $result.DirectoriesSkipped += $childResult.DirectoriesSkipped
            continue
        }

        Copy-Item -LiteralPath $child.FullName -Destination $targetPath -Force
        $result.FilesCopied++
    }

    return $result
}

$resolvedSourceRoot = (Resolve-Path -LiteralPath $SourceRoot).Path

if (-not (Test-Path -LiteralPath $DestinationRoot)) {
    $null = New-Item -ItemType Directory -Path $DestinationRoot -Force
}

Write-Host ("Source      : {0}" -f $resolvedSourceRoot) -ForegroundColor Cyan
Write-Host ("Destination : {0}" -f $DestinationRoot) -ForegroundColor Cyan
Write-Host ''

$items = Get-ChildItem -LiteralPath $resolvedSourceRoot -Force | Sort-Object -Property @{ Expression = { -not $_.PSIsContainer } }, Name
$topLevelDirectories = @($items | Where-Object { $_.PSIsContainer })
$eligibleDirectories = @($topLevelDirectories | Where-Object { $_.Name -notlike '*_partial*' })
$skippedDirectories = @($topLevelDirectories | Where-Object { $_.Name -like '*_partial*' })
$topLevelFiles = @($items | Where-Object { -not $_.PSIsContainer })

foreach ($skippedDirectory in $skippedDirectories) {
    Write-Host ("[SKIP] Top-level partial folder: {0}" -f $skippedDirectory.Name) -ForegroundColor Yellow
}

$folderIndex = 0
foreach ($directory in $eligibleDirectories) {
    $folderIndex++
    Write-Host ("[{0}/{1}] Copying folder: {2}" -f $folderIndex, $eligibleDirectories.Count, $directory.Name) -ForegroundColor Green

    $destinationPath = Join-Path -Path $DestinationRoot -ChildPath $directory.Name
    $copyResult = Copy-DirectoryTree -SourcePath $directory.FullName -DestinationPath $destinationPath

    Write-Host (
        "[DONE] {0} -> created {1} directories, copied {2} files, skipped {3} partial directories." -f
        $directory.Name,
        $copyResult.DirectoriesCreated,
        $copyResult.FilesCopied,
        $copyResult.DirectoriesSkipped
    ) -ForegroundColor Green
    Write-Host ''
}

foreach ($file in $topLevelFiles) {
    $targetPath = Join-Path -Path $DestinationRoot -ChildPath $file.Name
    Write-Host ("[FILE] Copying top-level file: {0}" -f $file.Name) -ForegroundColor Green
    Copy-Item -LiteralPath $file.FullName -Destination $targetPath -Force
}

Write-Host (
    "Completed copy of {0} folders and {1} top-level files into {2}." -f
    $eligibleDirectories.Count,
    $topLevelFiles.Count,
    $DestinationRoot
) -ForegroundColor Cyan
