#!/bin/sh
docker-compose -f docker-compose-deploy.yml build
docker-compose -f docker-compose-deploy.yml down -v
docker-compose -f docker-compose-deploy.yml up -d
