param(
    [Parameter(Mandatory = $true, Position = 0)]
    [string[]] $Path
)

$validationPattern = '^Validation (Error|Warning): \[ ([^\]]+) \]'

foreach ($inputPath in $Path) {
    foreach ($resolvedPath in Resolve-Path -Path $inputPath) {
        $messages = Select-String -LiteralPath $resolvedPath.Path -Pattern $validationPattern
        $identifiers = $messages | ForEach-Object { $_.Matches[0].Groups[2].Value }

        [pscustomobject]@{
            Log = $resolvedPath.Path
            ValidationMessages = @($messages).Count
            Bytes = (Get-Item -LiteralPath $resolvedPath.Path).Length
        } | Format-List

        $identifiers |
            Group-Object |
            Sort-Object Count -Descending |
            Format-Table Count, Name -AutoSize
    }
}
