FROM golang:1.25.3-alpine AS build
ARG MC_RELEASE=RELEASE.2025-08-13T08-35-41Z
RUN apk add --no-cache git && git clone --depth 1 --branch "$MC_RELEASE" https://github.com/minio/mc.git /src
WORKDIR /src
RUN CGO_ENABLED=0 go build -trimpath -o /out/mc .
FROM alpine:3.22.2
RUN apk add --no-cache ca-certificates
COPY --from=build /out/mc /usr/local/bin/mc
ENTRYPOINT ["/bin/sh"]
