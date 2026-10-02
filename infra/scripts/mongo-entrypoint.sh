#!/bin/sh
set -eu
cp /run/secrets/mongo_key /tmp/mongo-key
chown mongodb:mongodb /tmp/mongo-key
chmod 400 /tmp/mongo-key
exec /usr/local/bin/docker-entrypoint.sh mongod --replSet rs0 --keyFile /tmp/mongo-key --bind_ip "${MONGO_BIND_IP:-0.0.0.0}" --port "${MONGO_PORT:-27017}"
