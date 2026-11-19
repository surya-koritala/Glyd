#!/usr/bin/env python3
"""How much of a registry of popular container images is bytes shared
between images (public base layers): manifests from Docker Hub, no
pulls. Prints per-image bytes, the bytes in layers another image also
has, and the totals.  python3 public_bytes.py"""
import json, sys, urllib.request

IMAGES = """python:3.12 node:22 golang:1.23 ruby:3.3 php:8.3-apache eclipse-temurin:21 tomcat:10.1
nginx:latest httpd:latest caddy:latest traefik:latest haproxy:latest
redis:latest postgres:16 mysql:8 mariadb:11 mongo:7 elasticsearch:8.15.0 rabbitmq:3-management memcached:latest influxdb:2
wordpress:latest nextcloud:latest ghost:latest drupal:latest joomla:latest mediawiki:latest
grafana/grafana:latest jenkins/jenkins:lts sonarqube:community gitea/gitea:latest hashicorp/vault:latest hashicorp/consul:latest
prom/prometheus:latest minio/minio:latest registry:2 telegraf:latest
apache/airflow:latest apache/superset:latest metabase/metabase:latest bitnami/kafka:latest
alpine:latest ubuntu:24.04 debian:bookworm busybox:latest maven:3-eclipse-temurin-21 gradle:jdk21 composer:latest""".split()

ACCEPT = ", ".join(["application/vnd.docker.distribution.manifest.list.v2+json", "application/vnd.oci.image.index.v1+json",
                    "application/vnd.docker.distribution.manifest.v2+json", "application/vnd.oci.image.manifest.v1+json"])

def get(url, token=None, accept=None):
    req = urllib.request.Request(url)
    if token: req.add_header("Authorization", f"Bearer {token}")
    if accept: req.add_header("Accept", accept)
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())

def layers(image):
    repo, tag = image.split(":", 1)
    if "/" not in repo: repo = "library/" + repo
    token = get(f"https://auth.docker.io/token?service=registry.docker.io&scope=repository:{repo}:pull")["token"]
    m = get(f"https://registry-1.docker.io/v2/{repo}/manifests/{tag}", token, ACCEPT)
    if "manifests" in m:  # an index: the linux/amd64 image
        pick = [x for x in m["manifests"] if x.get("platform", {}).get("os") == "linux" and x["platform"].get("architecture") == "amd64"]
        if not pick: return []
        m = get(f"https://registry-1.docker.io/v2/{repo}/manifests/{pick[0]['digest']}", token, ACCEPT)
    return [(l["digest"], l["size"]) for l in m.get("layers", [])]

by_image = {}
for img in IMAGES:
    try:
        by_image[img] = layers(img)
    except Exception as e:
        print(f"{img}: {e}", file=sys.stderr)
holders = {}
for img, ls in by_image.items():
    for d, n in ls:
        holders.setdefault(d, set()).add(img)
total = shared = 0
print(f"{'image':38} {'MB':>8} {'shared MB':>10} {'shared':>7}")
for img, ls in by_image.items():
    t = sum(n for _, n in ls)
    s = sum(n for d, n in ls if len(holders[d]) > 1)
    total += t; shared += s
    print(f"{img:38} {t/1e6:8.1f} {s/1e6:10.1f} {100*s/max(t,1):6.0f}%")
unique = sum(next(n for i in holders[d] for dd, n in by_image[i] if dd == d) for d in holders)
print(f"\n{len(by_image)} images: {total/1e6:.0f} MB as stored one by one; {unique/1e6:.0f} MB of distinct layers;")
print(f"{100*shared/total:.0f}% of the bytes are in layers another image in the set also has; storing each layer once saves {100*(1-unique/total):.0f}%")
