#!/usr/bin/bash 

# abort on errors
set -e

# navigate into the docs directory
cd website

# build
npm run build
npx svelte-sitemap --domain https://www.world4ai.org 

# navigate into the build output directory
cd build

# if you are deploying to a custom domain
echo 'www.world4ai.org' > CNAME

touch .nojekyll

git init
git add -A
git commit -m 'deploy'

# if you are deploying to https://<USERNAME>.github.io/<REPO>
git push -f https://github.com/World4AI/World4AI.git master:gh-pages

cd -
