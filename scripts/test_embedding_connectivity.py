"""Standalone diagnostic script to test embedding endpoint reachability.

Run this from the project root. It will read `.env` (if present) and try a
small POST against a selection of likely endpoints so you can see exact
responses and debug connectivity.

Usage:
  python scripts/test_embedding_connectivity.py

This script tries a few candidate paths (base, /embeddings) and prints status
and a small portion of any response body.
"""
import os
import json
import textwrap
from dotenv import load_dotenv
import httpx


def short(s, n=400):
    s = s if isinstance(s, str) else json.dumps(s)
    return s if len(s) <= n else s[:n] + '...'


def get_env():
    load_dotenv()
    env = {
        'provider': os.getenv('EMBEDDING_PROVIDER'),
        'base': os.getenv('EMBEDDING_API_BASE'),
        'api_key': os.getenv('EMBEDDING_API_KEY'),
        'azure_base': os.getenv('AZURE_EMBEDDING_BASE'),
    }
    return env


async def try_post(url, key):
    headers = {'Content-Type': 'application/json'}
    if key:
        headers['Authorization'] = f'Bearer {key}'
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            r = await client.post(url, json={'input': 'ping'}, headers=headers)
            return r.status_code, r.text
    except Exception as e:
        return None, str(e)


async def main():
    env = get_env()
    print('Environment picks:')
    print(textwrap.indent(json.dumps(env, indent=2), '  '))

    if not env['base']:
        print('\nNo EMBEDDING_API_BASE configured. Set it in .env or environment and try again.')
        return

    cand = [env['base']]
    # add common variants
    if not env['base'].rstrip('/').endswith('/embeddings'):
        cand.append(env['base'].rstrip('/') + '/embeddings')
    cand.append(env['base'].rstrip('/') + '/v1/embeddings')

    print('\nTrying candidate endpoints:')
    from asyncio import gather
    tasks = [try_post(u, env['api_key']) for u in cand]
    res = await gather(*tasks)

    print('\nResults:')
    for u, (code, body) in zip(cand, res):
        print('---')
        print('URL:', u)
        if code is None:
            print('ERROR:', body)
        else:
            print('Status:', code)
            print('Body:', short(body))


if __name__ == '__main__':
    import asyncio
    asyncio.run(main())
