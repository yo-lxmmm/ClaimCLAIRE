# Vercel Deployment Setup Guide

## Environment Variables Required

Your Flask app needs these environment variables to work:

1. **GOOGLE_API_KEY** - Your Google Gemini API key
2. **SERPER_API_KEY** - Your Serper.dev API key (for web search)

## Setting Environment Variables

### Option 1: Via Vercel CLI (Recommended)

```bash
# Add GOOGLE_API_KEY for production
vercel env add GOOGLE_API_KEY production

# Add SERPER_API_KEY for production  
vercel env add SERPER_API_KEY production

# Also add for preview/development if needed
vercel env add GOOGLE_API_KEY preview
vercel env add SERPER_API_KEY preview
```

When prompted, paste your API key values.

### Option 2: Via Vercel Dashboard

1. Go to: https://vercel.com/yo-lxmmms-projects/inconsistency-detection/settings/environment-variables
2. Click "Add New"
3. Add each variable:
   - Name: `GOOGLE_API_KEY`, Value: `your-google-api-key`
   - Name: `SERPER_API_KEY`, Value: `your-serper-api-key`
4. Select "Production" (and optionally "Preview" and "Development")
5. Click "Save"

## After Setting Environment Variables

Redeploy your application:

```bash
vercel --prod
```

## Verify Deployment

1. Visit your production URL
2. Check the `/status` endpoint to verify the agent is initialized
3. Try analyzing a claim

## Troubleshooting

If you see a 500 error:
- Check that environment variables are set: `vercel env ls`
- Check deployment logs: `vercel logs <deployment-url>`
- Verify API keys are correct and have proper permissions

