
$API_HOST="127.0.0.1"
$API_KEY="0000000000000000000000000000000000000000000000000000000000000000"

$KEY_HEADER=@{
    'X-API-KEY' = $API_KEY
}

$node=Invoke-RestMethod -Uri "http://$($API_HOST):12973/infos/self-clique" -Method GET -Headers $KEY_HEADER -ErrorAction SilentlyContinue

if ($node -eq $null) {
	Write-Host "Your full node is not running"
	Exit 1
}

if ( ( -Not [bool]($node.PSobject.Properties.name -match "synced")) -or $node.synced -ne "True"){
	Write-Host "Your full node is not synced"
	Exit 1
}

$addresses=$(Invoke-RestMethod -Uri "http://$($API_HOST):12973/miners/addresses" -Method GET -Headers $KEY_HEADER -ErrorAction SilentlyContinue).addresses

if ($addresses -eq $null) {
    Write-Host "Miner addresses are not set"
    Exit 1
}

$miner_pid=$null
# Use try-finally to kill the miner automatically when the script stops (ctrl+c only)
try{
	while($true){
		# if we don't have an associated PID, spawn the miner and wait 10 seconds to ensure it started mining properly
		if ($miner_pid -eq $null){
			$miner_pid = (Start-Process "$PSScriptRoot\bin\gpu-miner.exe" "-a $($API_HOST)" -PassThru).ID
			Start-Sleep -Seconds 10
		}
		
		# Check if the process died
		if (-not (Get-Process -Id $miner_pid -ErrorAction SilentlyContinue)) {
			Write-Host "Miner died, restarting it..."
			$miner_pid=$null
			continue
		}
		
		# Check if GPU usage stalled (in some cases of error -4095, the process doesn't die, but GPU usage stops)
		$gpu_usage=0
		$usage_threshold=75 # Expect at least 75% gpu usage at all times
		((Get-Counter "\GPU Engine(pid_$($miner_pid)*engtype_Cuda)\Utilization Percentage").CounterSamples | where CookedValue).CookedValue |
			foreach { $gpu_usage = 0 } { $gpu_usage += [math]::Round($_,2) }
			
		Write-Output "Process $($miner_pid) GPU Engine Usage $($gpu_usage)%"
		
		if ($gpu_usage -lt $usage_threshold -and $miner_pid -ne $null){
			Write-Host "Miner stalled, restarting it"
			Stop-Process -Id $miner_pid -ErrorAction SilentlyContinue
			$miner_pid=$null
			continue
		}
		# Sleep 10 seconds before trying again
		Start-Sleep -Seconds 10
	}
}
finally
{
	if ($miner_pid -ne $null) {
		Stop-Process -Id $miner_pid -ErrorAction SilentlyContinue
	}
}
