# test_gitignore.ps1
# Tests whether specific files are being ignored by Git based on your .gitignore rules.

$filesToTest = @(
    ".test.ps1", 
    ".dev.ps1", 
    "test.ps1", 
    "dev.ps1"
)

Write-Host "Testing Git Ignore Rules..." -ForegroundColor Cyan
Write-Host "---------------------------"

foreach ($file in $filesToTest) {
    # Check if git ignores the file (returns 0 if ignored, 1 if not ignored)
    $ignoreRule = git check-ignore -v $file 2>$null
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "[IGNORED]     $file" -ForegroundColor Yellow
        Write-Host "              Matched rule: $ignoreRule" -ForegroundColor DarkGray
    } else {
        Write-Host "[NOT IGNORED] $file" -ForegroundColor Green
    }
}
Write-Host "---------------------------"
