#!/bin/sh
set -eu
# No tracing: credentials must not be printed.
mc alias set local http://minio:9000 "$MINIO_ROOT_USER" "$MINIO_ROOT_PASSWORD" >/dev/null
for bucket in kb-originals kb-derived kb-images milvus-internal; do
  mc mb --ignore-existing "local/$bucket" >/dev/null
done
cat >/tmp/business.json <<'EOF'
{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Action":["s3:ListBucket","s3:GetBucketLocation","s3:ListBucketVersions"],"Resource":["arn:aws:s3:::kb-originals","arn:aws:s3:::kb-derived","arn:aws:s3:::kb-images"]},{"Effect":"Allow","Action":["s3:GetObject","s3:PutObject","s3:DeleteObject","s3:GetObjectVersion","s3:DeleteObjectVersion"],"Resource":["arn:aws:s3:::kb-originals/*","arn:aws:s3:::kb-derived/*","arn:aws:s3:::kb-images/*"]}]}
EOF
cat >/tmp/query.json <<'EOF'
{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Action":["s3:ListBucket","s3:GetBucketLocation"],"Resource":["arn:aws:s3:::kb-originals","arn:aws:s3:::kb-derived","arn:aws:s3:::kb-images"]},{"Effect":"Allow","Action":["s3:GetObject"],"Resource":["arn:aws:s3:::kb-originals/*","arn:aws:s3:::kb-derived/*","arn:aws:s3:::kb-images/*"]}]}
EOF
cat >/tmp/milvus.json <<'EOF'
{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Action":["s3:*"],"Resource":["arn:aws:s3:::milvus-internal","arn:aws:s3:::milvus-internal/*"]}]}
EOF
mc admin policy create local kb-business /tmp/business.json >/dev/null
mc admin policy create local kb-query /tmp/query.json >/dev/null
mc admin policy create local milvus-storage /tmp/milvus.json >/dev/null
mc admin user add local "$MINIO_APP_USER" "$MINIO_APP_PASSWORD" >/dev/null
mc admin user add local "$MINIO_QUERY_USER" "$MINIO_QUERY_PASSWORD" >/dev/null
mc admin user add local "$MINIO_MILVUS_USER" "$MINIO_MILVUS_PASSWORD" >/dev/null
mc admin policy attach local kb-business --user "$MINIO_APP_USER" >/dev/null
mc admin policy attach local kb-query --user "$MINIO_QUERY_USER" >/dev/null
mc admin policy attach local milvus-storage --user "$MINIO_MILVUS_USER" >/dev/null
echo 'Private buckets and scoped storage accounts initialized.'
