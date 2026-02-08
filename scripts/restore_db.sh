#!/usr/bin/env bash
set -e

docker compose up -d

for i in {1..30}; do
  if docker exec guitarhero-mongo mongosh --quiet --eval 'db.runCommand({ping:1})' >/dev/null 2>&1; then
    break
  fi
  sleep 1
done

docker cp ./dump guitarhero-mongo:/dump
docker exec guitarhero-mongo mongorestore --drop /dump

echo "Restore completato. MongoDB su mongodb://localhost:27017"