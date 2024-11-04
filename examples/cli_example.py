"""
Command Line Interface Example
============================
Example of using AI Orchestrator in a CLI application.
"""

import asyncio
import argparse
from ai_orchestrator import AIOrchestrator
import json

async def process_file(file_path: str, api_key: str):
    orchestrator = AIOrchestrator(api_key=api_key)
    
    try:
        with open(file_path, 'r') as f:
            content = f.read()
            
        result = await orchestrator.process_input(
            session_id="cli-session",
            user_input=content
        )
        
        print(json.dumps(result, indent=2))
        
    except Exception as e:
        print(f"Error: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description='Process text using AI Orchestrator')
    parser.add_argument('file', help='Path to file to process')
    parser.add_argument('--api-key', required=True, help='API key for AI service')
    
    args = parser.parse_args()
    asyncio.run(process_file(args.file, args.api_key))

if __name__ == '__main__':
    main()