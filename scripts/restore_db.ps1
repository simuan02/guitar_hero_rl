# Avvia Mongo
docker compose up -d

$max = 30
for ($i=0; $i -lt $max; $i++) {
  try {
    docker exec guitarhero-mongo mongosh --quiet --eval "db.runCommand({ping:1})" | Out-Null
    break
  } catch {}
  Start-Sleep -Seconds 1
}

# Copia il dump dentro il container
docker cp .\dump guitarhero-mongo:/dump

# Ripristina
docker exec guitarhero-mongo mongorestore --drop /dump

Write-Host "Restore completato. MongoDB disponibile su mongodb://localhost:27017"