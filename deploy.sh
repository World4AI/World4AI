#!/usr/bin/env sh

# abort on errors
set -e

# navigate into the docs directory
cd docs

# build
npm run build

# navigate into the build output directory
cd build

# if you are deploying to a custom domain
echo 'www.world4ai.org' > CNAME

git init
git add -A
git commit -m 'deploy'

# if you are deploying to https://<USERNAME>.github.io/<REPO>
git push -f https://github.com/World4AI/World4AI.git master:gh-pages

cd -
