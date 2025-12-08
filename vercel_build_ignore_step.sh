#!/bin/bash

if [[ "$VERCEL_GIT_COMMIT_REF" == "main" ]] ; then
    # Proceed with the build if it's the main branch
    exit 1;
else
    # Don't build for other branches
    exit 0;
fi